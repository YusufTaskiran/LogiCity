import random

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from ..core.config import LABEL_MAP
from .observation_noise import apply_observation_noise


class SharedPolicyMultiAgentVecEnv(VecEnv):
    def __init__(self, env):
        self.env = env
        self.logic_grounding_shape = self.env.logic_grounding_shape
        self.pred_grounding_index = self.env.pred_grounding_index
        self.cat_length = env.rl_agent["cat_length"] if "cat_length" in env.rl_agent else False
        self.append_agent_id = bool(env.rl_agent.get("append_agent_id", False))
        self.agent_names = env.rl_agent["agent_names"]
        self.horizon = env.rl_agent["max_horizon"]
        self.action_mapping = env.rl_agent["action_mapping"]
        self.overtime_cost = env.rl_agent.get("overtime_cost", -3.0)
        self.reward_scheme = env.rl_agent.get("reward_scheme", "default")
        self.goal_reward = env.rl_agent.get("goal_reward", 0.0)
        self.progress_reward_scale = env.rl_agent.get("progress_reward_scale", 0.0)
        self.time_penalty = env.rl_agent.get("time_penalty", 0.0)
        self.reset_dist = env.rl_agent.get("reset_dist")
        self.reset_prefilter = env.rl_agent.get("reset_prefilter", None)
        self.observation_noise = env.rl_agent.get("observation_noise", None) or {}
        self.continuous_eval = bool((env.rl_agent.get("continuous_eval") or {}).get("enabled", False))
        self.max_priority = env.rl_agent["max_priority"]
        self.type2label = {v: k for k, v in LABEL_MAP.items()}

        self.rl_agents = []
        self.rl_layer_ids = []
        self.expert_action = None
        for agent_name in self.agent_names:
            agent_type, agent_id = agent_name.split("_")
            agent_id = int(agent_id)
            matched = None
            for agent in self.env.agents:
                if agent.type == agent_type and agent.id == agent_id:
                    matched = agent
                    break
            assert matched is not None, f"Agent {agent_name} not found for multi-agent RL."
            self.rl_agents.append(matched)
            self.rl_layer_ids.append(matched.layer_id)

        obs_dim = self.logic_grounding_shape
        if self.cat_length:
            obs_dim += 1
        if self.append_agent_id:
            obs_dim += 1
        observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        action_space = spaces.Discrete(env.rl_agent["action_space"])
        super().__init__(len(self.rl_agents), observation_space, action_space)

        self._actions = None
        self._completed_layer_ids = set()
        self._last_obs_batch = None
        self._last_distances = None
        self._agent_episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self._agent_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self._global_t = 0

    def full_action2index(self, action):
        if action[0] == 1:
            return 0
        elif action[4] == 1:
            return 1
        elif action[8] == 1:
            return 2
        else:
            return 3

    def _flatten_obs(self, obs_array, path_length, region, agent_batch_index):
        obs_array = self._apply_observation_noise(obs_array)
        features = [obs_array]
        if self.cat_length:
            normed_path_length = path_length / (2 * region)
            features.append(np.asarray([normed_path_length], dtype=np.float32))
        if self.append_agent_id:
            denom = max(self.num_envs - 1, 1)
            normed_agent_id = float(agent_batch_index) / float(denom)
            features.append(np.asarray([normed_agent_id], dtype=np.float32))
        if len(features) == 1:
            return obs_array
        return np.concatenate(features, axis=0, dtype=np.float32)

    def _apply_observation_noise(self, obs_array):
        return apply_observation_noise(obs_array, self.observation_noise, self.pred_grounding_index)

    def _trajectory_distance_remaining(self, agent):
        if agent.reach_goal:
            return 0.0
        traj = agent.global_traj
        pos = agent.pos
        matches = torch.all(traj == pos, dim=1).nonzero(as_tuple=False)
        if matches.numel() > 0:
            current_idx = int(matches[0].item())
            if current_idx >= len(traj) - 1:
                return 0.0
            deltas = traj[current_idx + 1:] - traj[current_idx:-1]
            return float(torch.abs(deltas).sum().item())
        return float(torch.abs(agent.goal - pos).sum().item())

    def _get_spf_reward(self, fail, sat_reward, prev_distance, curr_distance, reached_goal):
        if fail:
            return sat_reward
        progress = prev_distance - curr_distance
        reward = self.progress_reward_scale * progress + self.time_penalty
        if reached_goal:
            reward += self.goal_reward
        return reward

    def _get_center_intersection_mask(self, selector="center"):
        intersection_blocks = self.env.intersection_matrix[2]
        labels = torch.unique(intersection_blocks)
        labels = labels[labels > 0]
        if labels.numel() == 0:
            return None
        if selector != "center":
            raise ValueError(f"Unsupported intersection selector: {selector}")
        grid_center = torch.tensor(
            [intersection_blocks.shape[0] / 2.0, intersection_blocks.shape[1] / 2.0],
            dtype=torch.float32,
        )
        best_label = None
        best_distance = None
        for label in labels.tolist():
            coords = (intersection_blocks == label).nonzero(as_tuple=False).float()
            centroid = coords.mean(dim=0)
            distance = torch.norm(centroid - grid_center).item()
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_label = label
        return intersection_blocks == best_label

    def _agent_route_crosses_mask(self, agent, mask):
        traj = agent.global_traj
        if traj is None or len(traj) == 0:
            return False
        coords = traj.long()
        return bool(mask[coords[:, 0], coords[:, 1]].any().item())

    def _passes_reset_prefilter(self):
        if not self.reset_prefilter:
            return True
        selector = self.reset_prefilter.get("intersection_selector", "center")
        require_ego = self.reset_prefilter.get("require_ego_intersection", True)
        require_all = self.reset_prefilter.get("require_all_agents_intersection", False)
        min_other = self.reset_prefilter.get("min_other_intersection_agents", 1)
        mask = self._get_center_intersection_mask(selector=selector)
        if mask is None:
            return False
        rl_crosses = 0
        other_crosses = 0
        rl_layer_id_set = set(self.rl_layer_ids)
        for agent in self.env.agents:
            crosses = self._agent_route_crosses_mask(agent, mask)
            if agent.layer_id in rl_layer_id_set:
                if crosses:
                    rl_crosses += 1
                elif require_ego:
                    return False
            elif crosses:
                other_crosses += 1
            if require_all and not crosses:
                return False
        if require_ego and rl_crosses != len(self.rl_layer_ids):
            return False
        if other_crosses < min_other:
            return False
        return True

    def _redraw_all_agent_layers(self):
        for env_agent in self.env.agents:
            agent_code = self.type2label[env_agent.type]
            agent_layer = torch.zeros((self.env.grid_size[0], self.env.grid_size[1]))
            agent_layer[env_agent.start[0], env_agent.start[1]] = agent_code
            agent_layer[env_agent.goal[0], env_agent.goal[1]] = agent_code + 0.3
            for way_points in env_agent.global_traj[1:-1]:
                if torch.all(way_points == env_agent.start) or torch.all(way_points == env_agent.goal):
                    continue
                agent_layer[way_points[0], way_points[1]] = agent_code + 0.1
            agent_layer[env_agent.pos[0], env_agent.pos[1]] = agent_code
            self.env.city_grid[env_agent.layer_id] = agent_layer

    def _build_obs_batch(self, obs_dict):
        obs_batch = []
        distances = []
        for batch_index, (idx, agent) in enumerate(zip(self.rl_layer_ids, self.rl_agents)):
            path_length = len(agent.global_traj) * 4
            obs_batch.append(
                self._flatten_obs(obs_dict["World_state"][idx], path_length, agent.region, batch_index)
            )
            distances.append(self._trajectory_distance_remaining(agent))
        return np.stack(obs_batch, axis=0), np.array(distances, dtype=np.float32)

    def reset(self):
        max_attempts = 128
        for _ in range(max_attempts):
            self._global_t = 0
            self._completed_layer_ids = set()
            self._agent_episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
            self._agent_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
            for env_agent in self.env.agents:
                if getattr(env_agent, "cached_init_info", None) is not None:
                    env_agent.init(self.env.city_grid, init_info=env_agent.cached_init_info)
                else:
                    env_agent.init(self.env.city_grid)
                if env_agent.layer_id in self.rl_layer_ids and getattr(env_agent, "cached_init_info", None) is None:
                    env_agent.reset_concepts(self.max_priority, self.reset_dist)
            if self._passes_reset_prefilter():
                break
        else:
            raise RuntimeError("Failed to sample a multi-agent reset satisfying reset_prefilter.")

        self.env.local_planner.reset()
        self._redraw_all_agent_layers()
        obs_dict = self.env.update_multi(self.rl_layer_ids)
        self._last_obs_batch, self._last_distances = self._build_obs_batch(obs_dict)
        return self._last_obs_batch

    def init(self):
        return self.reset()

    def step_async(self, actions):
        self._actions = np.asarray(actions).copy()

    def step_wait(self):
        assert self._actions is not None
        self._global_t += 1

        actions_by_idx = {}
        for i, layer_id in enumerate(self.rl_layer_ids):
            if (not self.continuous_eval) and layer_id in self._completed_layer_ids:
                actions_by_idx[layer_id] = torch.tensor(self.action_mapping[3], dtype=torch.float32)
            else:
                action_id = int(self._actions[i])
                actions_by_idx[layer_id] = torch.tensor(self.action_mapping[action_id], dtype=torch.float32)

        prev_reach_goal = {agent.layer_id: bool(agent.reach_goal) for agent in self.rl_agents}
        move_info = self.env.move_rl_agents(actions_by_idx, completed_rl_agents=self._completed_layer_ids)
        fail_agent_layer_ids = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        agent_successes = np.zeros(self.num_envs, dtype=np.float32)
        goal_completion_layer_ids = []

        for i, agent in enumerate(self.rl_agents):
            layer_id = agent.layer_id
            per_agent = move_info["PerAgent"][layer_id]
            if (not self.continuous_eval) and layer_id in self._completed_layer_ids:
                rewards[i] = 0.0
                agent_successes[i] = 1.0
                continue
            curr_distance = self._trajectory_distance_remaining(agent)
            reward = self._get_spf_reward(
                per_agent["Fail"],
                per_agent["Reward"],
                float(self._last_distances[i]),
                curr_distance,
                bool(agent.reach_goal),
            )
            rewards[i] = reward
            self._agent_episode_rewards[i] += reward
            self._agent_episode_lengths[i] += 1
            if per_agent["Fail"]:
                fail_agent_layer_ids.append(layer_id)
            if agent.reach_goal and (not prev_reach_goal[layer_id]):
                goal_completion_layer_ids.append(layer_id)
            if agent.reach_goal and (not self.continuous_eval):
                self._completed_layer_ids.add(layer_id)
                agent_successes[i] = 1.0
            self._last_distances[i] = curr_distance

        joint_failure = len(fail_agent_layer_ids) > 0
        joint_success = (len(self._completed_layer_ids) == len(self.rl_layer_ids) and not joint_failure) if not self.continuous_eval else False
        # Continuous rollout still needs a hard horizon. Otherwise unresolved live goals can run indefinitely.
        timeout = (self._global_t >= self.horizon and not joint_success and not joint_failure)
        joint_done = joint_failure or joint_success or timeout

        obs_dict = self.env.update_multi(self.rl_layer_ids)
        obs_batch, distances = self._build_obs_batch(obs_dict)
        self._last_obs_batch = obs_batch
        self._last_distances = distances

        oracle_action_by_layer = {}
        for layer_id, action_tensor in obs_dict.get("Expert_actions", {}).items():
            oracle_action_by_layer[int(layer_id)] = int(self.full_action2index(action_tensor))

        stop_needed_layer_ids = [
            int(layer_id) for layer_id, action_id in oracle_action_by_layer.items() if action_id == 3
        ]

        infos = []
        for i, agent in enumerate(self.rl_agents):
            layer_id = agent.layer_id
            oracle_action = oracle_action_by_layer.get(int(layer_id), None)
            info = {
                "joint_is_success": joint_success,
                "joint_failure": joint_failure,
                "joint_timeout": timeout,
                "joint_done": joint_done,
                "agent_is_success": bool(layer_id in self._completed_layer_ids),
                "agent_reward": float(rewards[i]),
                "rl_agent_layer_id": layer_id,
                "rl_agent_name": self.agent_names[i],
                "num_rl_agents": self.num_envs,
                "mean_agent_success": float(agent_successes.mean()),
                "goal_completion_count": len(goal_completion_layer_ids),
                "goal_completion_layer_ids": list(goal_completion_layer_ids),
                "agent_goal_completed": bool(layer_id in goal_completion_layer_ids),
                "oracle_action": oracle_action,
                "stop_needed": bool(oracle_action == 3) if oracle_action is not None else False,
            }
            if layer_id in move_info["PerAgent"]:
                info["Fail"] = [bool(move_info["PerAgent"][layer_id]["Fail"])] if not joint_done else [False]
                info["FailRuleNames"] = [move_info["PerAgent"][layer_id]["FailRuleNames"]]
            else:
                info["Fail"] = [False]
                info["FailRuleNames"] = [[]]
            if i == 0:
                info["is_success"] = joint_success
                info["overtime"] = timeout
                info["joint_fail_agent_layer_ids"] = fail_agent_layer_ids
                info["num_stop_needed_step"] = len(stop_needed_layer_ids)
                info["stop_needed_layer_ids"] = stop_needed_layer_ids
            else:
                info["is_success"] = False
                info["overtime"] = False
            if joint_done:
                info["episode"] = {"r": float(self._agent_episode_rewards[i]), "l": int(self._agent_episode_lengths[i])}
            infos.append(info)

        dones = np.array([joint_done] * self.num_envs, dtype=bool)
        if joint_done:
            terminal_obs = obs_batch.copy()
            reset_obs = self.reset()
            for i in range(self.num_envs):
                infos[i]["terminal_observation"] = terminal_obs[i]
            obs_batch = reset_obs

        return obs_batch, rewards, dones, infos

    def close(self):
        return None

    def save_episode(self):
        city_grid = self.env.city_grid.clone()
        agents = {}
        for agent in self.env.agents:
            name = agent.type + "_" + str(agent.id)
            agents[name] = {
                "start": agent.start.clone().numpy(),
                "goal": agent.goal.clone().numpy(),
                "concepts": agent.concepts,
                "type": agent.type,
                "priority": agent.priority,
                "pos": agent.pos.clone().numpy(),
                "layer_id": agent.layer_id,
                "id": agent.id,
            }
        return {"city_grid": city_grid, "agents": agents}

    def get_attr(self, attr_name, indices=None):
        value = getattr(self, attr_name)
        return [value for _ in range(self.num_envs)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        method = getattr(self, method_name)
        result = method(*method_args, **method_kwargs)
        return [result for _ in range(self.num_envs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in range(self.num_envs)]

    def seed(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        return [seed for _ in range(self.num_envs)]
