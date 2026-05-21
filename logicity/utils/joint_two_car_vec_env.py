from __future__ import annotations

import copy
import logging
import pickle as pkl
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from logicity.shields import JointProbabilisticLogicShieldTwoCar
from logicity.utils.load import CityLoader
from logicity.utils.pred_converter.z3 import global_hard_failure_breakdown, global_stop_required_conflict

logger = logging.getLogger(__name__)


def build_city(simulation_config, episode_cache=None):
    city, cached_observation = CityLoader.from_yaml(**simulation_config, episode_cache=episode_cache)
    return city, cached_observation


class SharedPolicyTwoAgentRunner:
    def __init__(self, city, rl_agent_cfg, controlled_agent_names):
        self.city = city
        self.rl_agent_cfg = rl_agent_cfg
        self.controlled_agent_names = list(controlled_agent_names)
        self.cat_length = rl_agent_cfg.get("cat_length", False)
        self.grounding_mode = rl_agent_cfg.get("grounding_mode")
        self.action_mapping = rl_agent_cfg["action_mapping"]
        self.action_cost = rl_agent_cfg["action_cost"]
        self.num_actions = int(rl_agent_cfg["action_space"])
        self.stop_action_id = int(rl_agent_cfg.get("stop_action_id", self.num_actions - 1))
        self.horizon = int(rl_agent_cfg["max_horizon"])
        self.overtime_cost = float(rl_agent_cfg.get("overtime_cost", -3))
        self.controlled_agents = self._resolve_controlled_agents()
        self.controlled_layer_ids = {agent.layer_id for agent in self.controlled_agents}
        self.path_lengths = {
            agent.layer_id: max(len(agent.global_traj) * 4, 1) for agent in self.controlled_agents
        }
        self.normed_path_lengths = {
            agent.layer_id: len(agent.global_traj) / (2 * agent.region) for agent in self.controlled_agents
        }
        self.layer_id_to_name = {agent.layer_id: f"{agent.type}_{agent.id}" for agent in self.controlled_agents}
        self.layer_id_to_priority = {agent.layer_id: int(agent.priority) for agent in self.controlled_agents}
        self.t = 0
        self.parked_controlled_layers: set[int] = set()

    def _entity_name_from_layer(self, layer_id):
        for agent in self.city.agents:
            if int(agent.layer_id) == int(layer_id):
                return f"Entity_{agent.type}_{agent.layer_id}"
        raise KeyError(f"Agent layer {layer_id} not found.")

    def _active_agents_and_map(self):
        active_agents = []
        active_layerid2listid = {}
        for agent in self.city.agents:
            if agent.layer_id in self.parked_controlled_layers:
                continue
            active_layerid2listid[agent.layer_id] = len(active_agents)
            active_agents.append(agent)
        return active_agents, active_layerid2listid

    def _park_finished_controlled_agents(self):
        for agent in self.controlled_agents:
            if agent.reach_goal and agent.layer_id not in self.parked_controlled_layers:
                self.city.city_grid[agent.layer_id] *= 0
                self.parked_controlled_layers.add(agent.layer_id)

    @staticmethod
    def _path_to_set(path):
        return {tuple(int(v) for v in point.tolist()) for point in path}

    def _resolve_controlled_agents(self):
        selected = []
        for agent_name in self.controlled_agent_names:
            agent_type, agent_id = agent_name.split("_")
            match = None
            for agent in self.city.agents:
                if agent.type == agent_type and agent.id == int(agent_id):
                    match = agent
                    break
            if match is None:
                raise ValueError(f"Controlled agent {agent_name} not found in current city.")
            selected.append(match)
        return selected

    def _macro_action_index(self, action):
        if isinstance(action, (int, np.integer)):
            return int(action)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action)
        for idx, macro_action in self.action_mapping.items():
            macro_action = np.asarray(macro_action, dtype=np.float32)
            if action.shape == macro_action.shape and np.any((action > 0) & (macro_action > 0)):
                return int(idx)
        raise ValueError(f"Could not map action {action} to a discrete macro action.")

    def _flatten_obs(self, grounding, normed_path_length):
        if self.grounding_mode == "thesis_minimal":
            grounding = self._build_thesis_minimal_obs(grounding)
        if self.cat_length:
            return np.concatenate([grounding, [normed_path_length]], axis=0, dtype=np.float32)
        return grounding

    def _build_thesis_minimal_obs(self, grounding):
        entity_slots = int(self.rl_agent_cfg["fov_entities"]["Entity"])

        def _binary_idx(first_idx, second_idx):
            return first_idx * entity_slots + second_idx

        ego_in_inter = float(grounding.get("IsInInter_0", 0.0))
        other_in_inter = max(
            max(
                float(grounding.get(f"IsCar_{idx}", 0.0)),
                float(grounding.get(f"IsPedestrian_{idx}", 0.0)),
            )
            * float(grounding.get(f"IsInInter_{idx}", 0.0))
            for idx in range(1, entity_slots)
        )
        ego_at_inter = float(grounding.get("IsAtInter_0", 0.0))
        close_ahead = max(
            float(grounding.get(f"IsCloseAhead_{_binary_idx(0, idx)}", 0.0))
            for idx in range(1, entity_slots)
        )
        ahead = max(
            float(grounding.get(f"IsAhead_{_binary_idx(0, idx)}", 0.0))
            for idx in range(1, entity_slots)
        )
        higher_pri = max(
            max(
                float(grounding.get(f"IsCar_{idx}", 0.0)),
                float(grounding.get(f"IsPedestrian_{idx}", 0.0)),
            )
            * max(
                float(grounding.get(f"IsAtInter_{idx}", 0.0)),
                float(grounding.get(f"IsInInter_{idx}", 0.0)),
            )
            * float(grounding.get(f"HigherPri_{_binary_idx(idx, 0)}", 0.0))
            for idx in range(1, entity_slots)
        )

        return np.asarray(
            [ego_in_inter, other_in_inter, ego_at_inter, close_ahead, ahead, higher_pri],
            dtype=np.float32,
        )

    def _collect_agent_view(self, agent):
        active_agents, active_layerid2listid = self._active_agents_and_map()
        results = self.city.local_planner.plan(
            self.city.city_grid.clone(),
            self.city.intersection_matrix,
            active_agents,
            active_layerid2listid,
            use_multiprocessing=self.city.use_multi,
            rl_agent=agent.layer_id,
        )
        agent_key = f"{agent.type}_{agent.layer_id}"
        grounding = results[f"{agent_key}_grounding"]
        grounding_dic = results.get(f"{agent_key}_grounding_dic")
        expert_action = results.get(f"{agent_key}_action")
        expert_idx = None
        if expert_action is not None:
            expert_idx = self._macro_action_index(expert_action)
        obs_grounding = grounding_dic if (self.grounding_mode == "thesis_minimal" and grounding_dic is not None) else grounding
        return {
            "obs": self._flatten_obs(obs_grounding, self.normed_path_lengths[agent.layer_id]),
            "grounding": grounding,
            "grounding_dic": grounding_dic,
            "expert_action": expert_idx,
        }

    def collect_views(self):
        self.city.local_planner.reset()
        return {
            agent.layer_id: self._collect_agent_view(agent)
            for agent in self.controlled_agents
            if agent.layer_id not in self.parked_controlled_layers
        }

    def passes_quality_filter(self):
        require_ped_route_intersection = bool(self.rl_agent_cfg.get("require_ped_route_intersection", False))
        require_car_route_intersection = bool(self.rl_agent_cfg.get("require_car_route_intersection", False))
        ped_agents = [agent for agent in self.city.agents if agent.type == "Pedestrian"]
        if require_ped_route_intersection and ped_agents:
            ped_paths = [self._path_to_set(agent.global_traj) for agent in ped_agents]
            for agent in self.controlled_agents:
                car_path = self._path_to_set(agent.global_traj)
                if not any(car_path.intersection(ped_path) for ped_path in ped_paths):
                    return False
        if require_car_route_intersection and len(self.controlled_agents) >= 2:
            shared = self._path_to_set(self.controlled_agents[0].global_traj).intersection(
                self._path_to_set(self.controlled_agents[1].global_traj)
            )
            if len(shared) == 0:
                return False
        return True

    def _reward_for_action(self, agent, grounding, discrete_action):
        one_hot_action = torch.tensor(self.action_mapping[int(discrete_action)], dtype=torch.float32)
        fail, sat_reward = self.city.local_planner.eval_state_action(grounding, one_hot_action)
        if fail:
            return float(sat_reward), bool(fail)
        moving_cost = self.action_cost[int(discrete_action)]
        return float((moving_cost + sat_reward) / self.path_lengths[agent.layer_id]), bool(fail)

    def _move_controlled_agents(self, actions):
        for agent in self.controlled_agents:
            if agent.reach_goal:
                continue
            discrete_action = int(actions[agent.layer_id])
            one_hot_action = torch.tensor(self.action_mapping[discrete_action], dtype=torch.float32)
            local_action, next_layer = agent.get_next_action(self.city.city_grid, one_hot_action)
            if not agent.reach_goal:
                next_layer = agent.move(local_action, next_layer)
            self.city.city_grid[agent.layer_id] = next_layer
        self._park_finished_controlled_agents()

    def _move_uncontrolled_agents(self):
        active_agents, active_layerid2listid = self._active_agents_and_map()
        agent_action_dist = self.city.local_planner.plan(
            self.city.city_grid.clone(),
            self.city.intersection_matrix,
            active_agents,
            active_layerid2listid,
            use_multiprocessing=self.city.use_multi,
            rl_agent=None,
        )
        new_layers = {}
        for agent in active_agents:
            if agent.layer_id in self.controlled_layer_ids:
                continue
            agent_key = f"{agent.type}_{agent.layer_id}"
            local_action_dist = agent_action_dist[agent_key]
            local_action, next_layer = agent.get_next_action(self.city.city_grid, local_action_dist)
            if not agent.reach_goal:
                next_layer = agent.move(local_action, next_layer)
            new_layers[agent.layer_id] = next_layer
        for layer_id, next_layer in new_layers.items():
            self.city.city_grid[layer_id] = next_layer

    def step(self, current_views, actions):
        self.t += 1
        prev_world = self.city.city_grid.clone()
        rewards = {}
        fails = {}
        rule_based_fail_count = 0
        for agent in self.controlled_agents:
            layer_id = agent.layer_id
            if layer_id in self.parked_controlled_layers:
                rewards[layer_id] = 0.0
                fails[layer_id] = False
                continue
            reward, fail = self._reward_for_action(agent, current_views[layer_id]["grounding"], actions[layer_id])
            ego_entity = self._entity_name_from_layer(layer_id)
            active_agents, _ = self._active_agents_and_map()
            stop_required = int(
                global_stop_required_conflict(
                    prev_world,
                    self.city.intersection_matrix,
                    active_agents,
                    ego_entity,
                )
            )
            rule_based_fail = int(stop_required and int(actions[layer_id]) != self.stop_action_id)
            if rule_based_fail > 0:
                reward += -10 * rule_based_fail
                fail = True
                rule_based_fail_count += rule_based_fail
            rewards[layer_id] = reward
            fails[layer_id] = fail

        self._move_controlled_agents(actions)
        self._move_uncontrolled_agents()

        active_agents, _ = self._active_agents_and_map()
        hard_breakdown = global_hard_failure_breakdown(
            prev_world,
            self.city.city_grid,
            self.city.intersection_matrix,
            active_agents,
        )
        deadzone_fail_count = int(hard_breakdown["deadzone"])
        simultaneous_entry_fail_count = int(hard_breakdown["simultaneous_entry"])
        event_hard_fail_count = deadzone_fail_count + simultaneous_entry_fail_count
        if event_hard_fail_count > 0:
            for layer_id in rewards:
                rewards[layer_id] += -10 * event_hard_fail_count

        overtime = self.t >= self.horizon
        all_success = all(agent.reach_goal for agent in self.controlled_agents)
        any_fail = any(fails.values()) or (event_hard_fail_count > 0)
        done = all_success or any_fail or overtime
        if overtime:
            for layer_id in rewards:
                rewards[layer_id] += self.overtime_cost

        next_views = self.collect_views() if not done else {}
        info = {
            "is_success": all_success and not any_fail and not overtime,
            "overtime": overtime,
            "any_fail": any_fail,
            "rule_based_fail_count": int(rule_based_fail_count),
            "deadzone_fail_count": int(deadzone_fail_count),
            "simultaneous_entry_fail_count": int(simultaneous_entry_fail_count),
            "agent_fail": {self.layer_id_to_name[k]: bool(v) for k, v in fails.items()},
            "agent_success": {
                self.layer_id_to_name[agent.layer_id]: bool(agent.reach_goal) for agent in self.controlled_agents
            },
            "World": self.city.city_grid.clone(),
        }
        return next_views, rewards, done, info


def _resolve_controlled_agent_names(rl_agent_cfg):
    controlled_agent_names = rl_agent_cfg.get("agent_names")
    if controlled_agent_names:
        return list(controlled_agent_names)
    single_agent_name = rl_agent_cfg.get("agent_name")
    if single_agent_name == "Car_1":
        return ["Car_1", "Car_2"]
    raise ValueError("Shared joint training requires simulation.rl_agent.agent_names or an agent_name compatible with the default two-car setup.")


def _ego_at_inter_from_view(view: dict[str, Any] | None) -> bool:
    if not view:
        return False
    obs = np.asarray(view.get("obs", []), dtype=np.float32).reshape(-1)
    if obs.shape[0] < 3:
        return False
    return bool(float(obs[2]) > 0.5)


class JointTwoCarVecEnv(VecEnv):
    def __init__(self, simulation_config: dict[str, Any], shield_config: dict[str, Any] | None = None, episode_data_path: str | None = None, seed: int = 2):
        self.simulation_config = copy.deepcopy(simulation_config)
        self.rl_agent_cfg = self.simulation_config["rl_agent"]
        self.controlled_agent_names = _resolve_controlled_agent_names(self.rl_agent_cfg)
        if len(self.controlled_agent_names) != 2:
            raise ValueError("Joint two-car training requires exactly two controlled agents.")
        self.shield_config = copy.deepcopy(shield_config or {})
        self.render_mode = None
        self._rng = np.random.default_rng(seed)
        self._pending_actions = None
        self._episode_index = 0
        self._cached_episodes = None
        if episode_data_path:
            with open(episode_data_path, "rb") as f:
                self._cached_episodes = list(pkl.load(f).values())

        bootstrap_obs = self._reset_joint_episode()
        obs_dim = int(bootstrap_obs.shape[1])
        observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        action_space = spaces.Discrete(int(self.rl_agent_cfg["action_space"]))
        super().__init__(num_envs=2, observation_space=observation_space, action_space=action_space)
        self.pred_grounding_index = copy.deepcopy(self.city.pred_grounding_index)
        self.current_shield_context = None
        self._reset_infos()

    def _reset_infos(self):
        self.reset_infos = [
            {"agent_name": self.controlled_agent_names[0]},
            {"agent_name": self.controlled_agent_names[1]},
        ]

    def _sample_episode_cache(self):
        if not self._cached_episodes:
            return None
        idx = int(self._rng.integers(0, len(self._cached_episodes)))
        return copy.deepcopy(self._cached_episodes[idx])

    def _build_runner(self, episode_cache=None):
        controlled_agent_names = self.controlled_agent_names
        attempts = int(self.rl_agent_cfg.get("sample_attempts", 1))
        last_city = None
        last_cache = None
        for _ in range(attempts):
            city, cached_observation = build_city(self.simulation_config, episode_cache=episode_cache)
            runner = SharedPolicyTwoAgentRunner(city, self.rl_agent_cfg, controlled_agent_names)
            if episode_cache is not None or runner.passes_quality_filter():
                return city, cached_observation, runner
            last_city = city
            last_cache = cached_observation
        runner = SharedPolicyTwoAgentRunner(last_city, self.rl_agent_cfg, controlled_agent_names)
        return last_city, last_cache, runner

    def _obs_batch_from_views(self, views):
        obs_batch = []
        for agent in self.runner.controlled_agents:
            layer_id = agent.layer_id
            if layer_id in views:
                obs_batch.append(np.asarray(views[layer_id]["obs"], dtype=np.float32))
            else:
                obs_batch.append(np.zeros(self.observation_space.shape, dtype=np.float32) if hasattr(self, "observation_space") else np.zeros(6, dtype=np.float32))
        return np.stack(obs_batch, axis=0).astype(np.float32)

    def _reset_joint_episode(self):
        episode_cache = self._sample_episode_cache()
        self.city, self.cached_observation, self.runner = self._build_runner(episode_cache=episode_cache)
        self.current_views = self.runner.collect_views()
        self._current_obs = self._obs_batch_from_views(self.current_views)
        self.pred_grounding_index = copy.deepcopy(self.city.pred_grounding_index)
        self._episode_reward = np.zeros(2, dtype=np.float32)
        self._episode_length = 0
        self._current_context_cache = None
        return self._current_obs.copy()

    def _joint_facts_and_mask(self):
        if self._current_context_cache is not None:
            return self._current_context_cache
        active_agents = [agent for agent in self.runner.controlled_agents if agent.layer_id not in self.runner.parked_controlled_layers]
        # Joint PLS is purely observation/regime-based: no per-pair simulator probing.
        candidate_infos = {}
        admissible_mask = np.ones((self.action_space.n, self.action_space.n), dtype=np.float32)

        if len(active_agents) == 2:
            both_at_inter = bool(
                _ego_at_inter_from_view(self.current_views.get(active_agents[0].layer_id))
                and _ego_at_inter_from_view(self.current_views.get(active_agents[1].layer_id))
            )
            same_priority = bool(int(active_agents[0].priority) == int(active_agents[1].priority))
            tie_break_a = bool(int(active_agents[0].layer_id) <= int(active_agents[1].layer_id))
        else:
            both_at_inter = False
            same_priority = False
            tie_break_a = True
            parked_index = 0 if self.runner.controlled_agents[0].layer_id in self.runner.parked_controlled_layers else 1
            admissible_mask[:] = 0.0
            if parked_index == 0:
                admissible_mask[self.runner.stop_action_id, :] = 1.0
            else:
                admissible_mask[:, self.runner.stop_action_id] = 1.0

        self._current_context_cache = {
            "candidate_infos": candidate_infos,
            "admissible_mask": admissible_mask,
            "both_at_inter": both_at_inter,
            "same_priority": same_priority,
            "tie_break_a": tie_break_a,
        }
        return self._current_context_cache

    def current_joint_context(self):
        return copy.deepcopy(self._joint_facts_and_mask())

    def reset(self):
        obs = self._reset_joint_episode()
        self._reset_seeds()
        self._reset_options()
        self._reset_infos()
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = np.asarray(actions, dtype=np.int64).reshape(self.num_envs)

    def step_wait(self):
        if self._pending_actions is None:
            raise RuntimeError("step_async() must be called before step_wait().")
        actions = self._pending_actions
        self._pending_actions = None

        action_map = {
            self.runner.controlled_agents[0].layer_id: int(actions[0]),
            self.runner.controlled_agents[1].layer_id: int(actions[1]),
        }
        next_views, reward_map, done, info = self.runner.step(self.current_views, action_map)
        rewards = np.asarray(
            [
                float(reward_map[self.runner.controlled_agents[0].layer_id]),
                float(reward_map[self.runner.controlled_agents[1].layer_id]),
            ],
            dtype=np.float32,
        )
        self._episode_reward += rewards
        self._episode_length += 1

        if done:
            terminal_obs = self._obs_batch_from_views(next_views)
            episode_info = {
                "r": float(np.mean(self._episode_reward)),
                "l": int(self._episode_length),
                "joint_success": bool(info.get("is_success", False)),
            }
            infos = [
                {
                    "episode": episode_info,
                    "terminal_observation": terminal_obs[idx],
                    "TimeLimit.truncated": bool(info.get("overtime", False)),
                    "joint_success": bool(info.get("is_success", False)),
                }
                for idx in range(self.num_envs)
            ]
            new_obs = self._reset_joint_episode()
            dones = np.array([True, True], dtype=bool)
            return new_obs, rewards, dones, infos

        self.current_views = next_views
        self._current_obs = self._obs_batch_from_views(self.current_views)
        self._current_context_cache = None
        infos = [
            {
                "joint_success": False,
                "rule_based_fail_count": int(info.get("rule_based_fail_count", 0)),
                "deadzone_fail_count": int(info.get("deadzone_fail_count", 0)),
                "simultaneous_entry_fail_count": int(info.get("simultaneous_entry_fail_count", 0)),
            }
            for _ in range(self.num_envs)
        ]
        dones = np.array([False, False], dtype=bool)
        return self._current_obs.copy(), rewards, dones, infos

    def close(self) -> None:
        return None

    def get_attr(self, attr_name: str, indices=None):
        values = []
        for _ in self._get_indices(indices):
            values.append(getattr(self, attr_name))
        return values

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        method = getattr(self, method_name)
        return [method(*method_args, **method_kwargs) for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices=None):
        return [False for _ in self._get_indices(indices)]

    def get_images(self):
        return [None for _ in range(self.num_envs)]


class CentralizedJointTwoCarEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        simulation_config: dict[str, Any],
        shield_config: dict[str, Any] | None = None,
        episode_data_path: str | None = None,
        fixed_episode_cache: dict[str, Any] | None = None,
        seed: int = 2,
    ):
        super().__init__()
        self.simulation_config = copy.deepcopy(simulation_config)
        self.rl_agent_cfg = self.simulation_config["rl_agent"]
        self.controlled_agent_names = _resolve_controlled_agent_names(self.rl_agent_cfg)
        if len(self.controlled_agent_names) != 2:
            raise ValueError("Centralized joint training requires exactly two controlled agents.")
        self.shield_config = copy.deepcopy(shield_config or {})
        self.render_mode = None
        self._rng = np.random.default_rng(seed)
        self._cached_episodes = None
        self._fixed_episode_cache = copy.deepcopy(fixed_episode_cache) if fixed_episode_cache is not None else None
        if episode_data_path:
            with open(episode_data_path, "rb") as f:
                self._cached_episodes = list(pkl.load(f).values())

        self.local_obs_dim = 6 + (1 if bool(self.rl_agent_cfg.get("cat_length", False)) else 0)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.local_obs_dim * 2,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(int(self.rl_agent_cfg["action_space"]) ** 2)
        self.num_joint_actions = int(self.rl_agent_cfg["action_space"]) ** 2
        action_labels = self.shield_config.get("action_space", ["fast", "normal", "slow", "stop"])
        self._joint_selector = JointProbabilisticLogicShieldTwoCar(
            num_actions=int(self.rl_agent_cfg["action_space"]),
            stop_action=int(self.rl_agent_cfg.get("stop_action_id", int(self.rl_agent_cfg["action_space"]) - 1)),
            action_space=action_labels,
            graded_safety_weights=self.shield_config.get("graded_safety_weights"),
        )
        self.pred_grounding_index = None
        self.current_shield_context = None
        self._episode_reward = 0.0
        self._episode_length = 0
        self._reset_state()

    def _sample_episode_cache(self):
        if self._fixed_episode_cache is not None:
            return copy.deepcopy(self._fixed_episode_cache)
        if not self._cached_episodes:
            return None
        idx = int(self._rng.integers(0, len(self._cached_episodes)))
        return copy.deepcopy(self._cached_episodes[idx])

    def _build_runner(self, episode_cache=None):
        attempts = int(self.rl_agent_cfg.get("sample_attempts", 1))
        last_city = None
        last_cache = None
        for _ in range(attempts):
            city, cached_observation = build_city(self.simulation_config, episode_cache=episode_cache)
            runner = SharedPolicyTwoAgentRunner(city, self.rl_agent_cfg, self.controlled_agent_names)
            if episode_cache is not None or runner.passes_quality_filter():
                return city, cached_observation, runner
            last_city = city
            last_cache = cached_observation
        runner = SharedPolicyTwoAgentRunner(last_city, self.rl_agent_cfg, self.controlled_agent_names)
        return last_city, last_cache, runner

    def _obs_from_views(self, views):
        obs_parts = []
        for agent in self.runner.controlled_agents:
            layer_id = agent.layer_id
            if layer_id in views:
                obs_parts.append(np.asarray(views[layer_id]["obs"], dtype=np.float32))
            else:
                obs_parts.append(np.zeros((self.local_obs_dim,), dtype=np.float32))
        return np.concatenate(obs_parts, axis=0, dtype=np.float32)

    def _joint_facts_and_mask(self):
        active_agents = [agent for agent in self.runner.controlled_agents if agent.layer_id not in self.runner.parked_controlled_layers]
        # Centralized joint PLS uses fixed regime safety scores only.
        candidate_infos = {}
        admissible_mask = np.ones((self.runner.num_actions, self.runner.num_actions), dtype=np.float32)

        if len(active_agents) == 2:
            both_at_inter = bool(
                _ego_at_inter_from_view(self.current_views.get(active_agents[0].layer_id))
                and _ego_at_inter_from_view(self.current_views.get(active_agents[1].layer_id))
            )
            same_priority = bool(int(active_agents[0].priority) == int(active_agents[1].priority))
            tie_break_a = bool(int(active_agents[0].layer_id) <= int(active_agents[1].layer_id))
            tiebreak_order = (int(active_agents[0].layer_id), int(active_agents[1].layer_id))
        else:
            both_at_inter = False
            same_priority = False
            tie_break_a = True
            tiebreak_order = (
                int(self.runner.controlled_agents[0].layer_id),
                int(self.runner.controlled_agents[1].layer_id),
            )
            parked_index = 0 if self.runner.controlled_agents[0].layer_id in self.runner.parked_controlled_layers else 1
            admissible_mask[:] = 0.0
            if parked_index == 0:
                admissible_mask[self.runner.stop_action_id, :] = 1.0
            else:
                admissible_mask[:, self.runner.stop_action_id] = 1.0

        safe_scores = self._joint_selector.joint_safe_scores(
            candidate_infos=candidate_infos,
            joint_facts={"both_at_inter": both_at_inter, "same_priority": same_priority},
            tiebreak_order=tiebreak_order,
        )
        return {
            "candidate_infos": candidate_infos,
            "admissible_mask": admissible_mask,
            "both_at_inter": both_at_inter,
            "same_priority": same_priority,
            "tie_break_a": tie_break_a,
            "tiebreak_order": tiebreak_order,
            "safe_scores": safe_scores,
        }

    def _refresh_context(self):
        self.current_shield_context = self._joint_facts_and_mask()

    def _reset_state(self):
        episode_cache = self._sample_episode_cache()
        self.city, self.cached_observation, self.runner = self._build_runner(episode_cache=episode_cache)
        self.current_views = self.runner.collect_views()
        self.pred_grounding_index = copy.deepcopy(self.city.pred_grounding_index)
        self._episode_reward = 0.0
        self._episode_length = 0
        self._refresh_context()
        self._last_obs = self._obs_from_views(self.current_views)
        return self._last_obs.copy()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        super().reset(seed=seed)
        obs = self._reset_state()
        return obs, {}

    def _decode_joint_action(self, joint_action: int) -> tuple[int, int]:
        joint_action = int(joint_action)
        num_actions = self.runner.num_actions
        return joint_action // num_actions, joint_action % num_actions

    def step(self, action):
        a0, a1 = self._decode_joint_action(int(action))
        action_map = {
            self.runner.controlled_agents[0].layer_id: int(a0),
            self.runner.controlled_agents[1].layer_id: int(a1),
        }
        next_views, reward_map, done, info = self.runner.step(self.current_views, action_map)
        rewards = [
            float(reward_map[self.runner.controlled_agents[0].layer_id]),
            float(reward_map[self.runner.controlled_agents[1].layer_id]),
        ]
        reward = float(np.mean(rewards))
        self._episode_reward += reward
        self._episode_length += 1

        if done:
            terminal_obs = self._obs_from_views(next_views)
            episode_info = {
                "r": float(self._episode_reward),
                "l": int(self._episode_length),
                "joint_success": bool(info.get("is_success", False)),
            }
            step_info = {
                "episode": episode_info,
                "terminal_observation": terminal_obs,
                "TimeLimit.truncated": bool(info.get("overtime", False)),
                "joint_success": bool(info.get("is_success", False)),
                "any_fail": bool(info.get("any_fail", False)),
                "overtime": bool(info.get("overtime", False)),
                "rule_based_fail_count": int(info.get("rule_based_fail_count", 0)),
                "deadzone_fail_count": int(info.get("deadzone_fail_count", 0)),
                "simultaneous_entry_fail_count": int(info.get("simultaneous_entry_fail_count", 0)),
                "agent_success": dict(info.get("agent_success", {})),
            }
            reset_obs = self._reset_state()
            return reset_obs, reward, True, False, step_info

        self.current_views = next_views
        self._refresh_context()
        self._last_obs = self._obs_from_views(self.current_views)
        step_info = {
            "joint_success": False,
            "rule_based_fail_count": int(info.get("rule_based_fail_count", 0)),
            "deadzone_fail_count": int(info.get("deadzone_fail_count", 0)),
            "simultaneous_entry_fail_count": int(info.get("simultaneous_entry_fail_count", 0)),
        }
        return self._last_obs.copy(), reward, False, False, step_info
