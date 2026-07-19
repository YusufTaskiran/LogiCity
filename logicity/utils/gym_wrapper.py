import gymnasium as gym
import torch
import numpy as np
from gymnasium.spaces import Box, Dict
import torch.nn.functional as F
from ..core.config import *
from .observation_noise import apply_observation_noise

import logging
import random
logger = logging.getLogger(__name__)
_RESET_SENTINEL = object()

def CPU(x):
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

def CUDA(x):
    return x.cuda() if isinstance(x, torch.Tensor) else x

class GymCityWrapper(gym.core.Env):
    def __init__(self, env):
        '''The Gym Wrapper of the CityEnv in single-agent mode.
        :param City env: the CityEnv instance
        '''        
        self.env = env
        self.logic_grounding_shape = self.env.logic_grounding_shape
        self.pred_grounding_index = self.env.pred_grounding_index
        # self.observation_space = Dict({
        #     "map": Box(low=-1.0, high=1.0, shape=(3, self.fov, self.fov), dtype=np.float32),  # Adjust the shape as needed
        #     "position": Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        # })
        self.cat_length = env.rl_agent["cat_length"] if "cat_length" in env.rl_agent else False
        if self.cat_length:
            self.observation_space = Box(low=0.0, high=1.0, shape=(self.logic_grounding_shape + 1, ), dtype=np.float32)
        else:
            self.observation_space = Box(low=0.0, high=1.0, shape=(self.logic_grounding_shape, ), dtype=np.float32)
        self.last_dist = -1
        self.agent_name = env.rl_agent["agent_name"]
        self.horizon = env.rl_agent["max_horizon"]
        self.agent_type = self.agent_name.split("_")[0]
        agent_id = self.agent_name.split("_")[1] # this is agent id in the yaml file
        self.use_expert = env.rl_agent["use_expert"]
        for agent in self.env.agents:
            if agent.type == self.agent_type:
                if agent.id == int(agent_id):
                    self.agent = agent
                    self.agent_layer_id = agent.layer_id
        assert self.agent_layer_id is not None, "Agent not found! Recheck Your agent_name in the config file!"
        action_space = self.env.rl_agent["action_space"]
        if self.use_expert:
            self.expert_action = np.zeros(action_space, dtype=np.float32)
        else:
            self.expert_action = -1
        self.action_space = gym.spaces.Discrete(action_space)
        self.action_mapping = env.rl_agent["action_mapping"]
        self.max_priority = env.rl_agent["max_priority"]
        self.action_cost = env.rl_agent["action_cost"]
        self.reset_dist = env.rl_agent["reset_dist"] if "reset_dist" in env.rl_agent else None
        self.overtime_cost = env.rl_agent["overtime_cost"] if "overtime_cost" in env.rl_agent else -3
        self.reward_scheme = env.rl_agent.get("reward_scheme", "default")
        self.goal_reward = env.rl_agent.get("goal_reward", 0.0)
        self.progress_reward_scale = env.rl_agent.get("progress_reward_scale", 0.0)
        self.time_penalty = env.rl_agent.get("time_penalty", 0.0)
        self.reset_prefilter = env.rl_agent.get("reset_prefilter", None)
        self.observation_noise = env.rl_agent.get("observation_noise", None) or {}
        self.type2label = {v: k for k, v in LABEL_MAP.items()}
        self.scale = [25, 7, 3.5, 8.3]
        self.mini_scale = [0, 0, -1, 0]
        self.t = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.last_planner_actions = {}
        self.last_agent_positions = {}

    def full_action2index(self, action):
        # see agents/car.py
        if action[0] == 1:
            return 0
        elif action[4] == 1:
            return 1
        elif action[8] == 1:
            return 2
        else:
            return 3
    
    def _flatten_obs(self, obs_dict):
        if self.cat_length:
            logic_obs = self._apply_observation_noise(obs_dict["World_state"][0])
            return np.concatenate([logic_obs, [self.normed_path_length]], axis=0, dtype=np.float32)
        else:
            return self._apply_observation_noise(obs_dict["World_state"][0])

    def _apply_observation_noise(self, obs_array):
        return apply_observation_noise(obs_array, self.observation_noise, self.pred_grounding_index)
                
    def _get_reward(self, obs_dict):
        ''' Get the reward for the current step.
        :param dict obs_dict: the observation dictionary
        :return: the reward
        '''
        if obs_dict["Fail"][0]:
            # failing step do not normailze the reward
            return obs_dict["Reward"][0]
        else:
            moving_cost = self.action2cost(obs_dict["Agent_actions"][0])
            return (moving_cost + obs_dict["Reward"][0])/self.path_length

    def _trajectory_distance_remaining(self):
        if self.agent.reach_goal:
            return 0.0
        traj = self.agent.global_traj
        pos = self.agent.pos
        matches = torch.all(traj == pos, dim=1).nonzero(as_tuple=False)
        if matches.numel() > 0:
            current_idx = int(matches[0].item())
            if current_idx >= len(traj) - 1:
                return 0.0
            deltas = traj[current_idx + 1:] - traj[current_idx:-1]
            return float(torch.abs(deltas).sum().item())
        return float(torch.abs(self.agent.goal - pos).sum().item())

    def _get_spf_reward(self, obs_dict, prev_distance, curr_distance, reached_goal):
        if obs_dict["Fail"][0]:
            return obs_dict["Reward"][0]
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
            raise ValueError("Unsupported intersection selector: {}".format(selector))
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
        ego_crosses = False
        other_crosses = 0
        for agent in self.env.agents:
            crosses = self._agent_route_crosses_mask(agent, mask)
            if agent.layer_id == self.agent_layer_id:
                ego_crosses = crosses
            elif crosses:
                other_crosses += 1
            if require_all and not crosses:
                return False
        if require_ego and not ego_crosses:
            return False
        if other_crosses < min_other:
            return False
        return True

    def _redraw_all_agent_layers(self):
        for env_agent in self.env.agents:
            agent_code = self.type2label[env_agent.type]
            agent_layer = torch.zeros((self.env.grid_size[0], self.env.grid_size[1]))
            agent_layer[env_agent.start[0], env_agent.start[1]] = agent_code
            agent_layer[env_agent.goal[0], env_agent.goal[1]] = agent_code + AGENT_GOAL_PLUS
            for way_points in env_agent.global_traj[1:-1]:
                if torch.all(way_points == env_agent.start) or torch.all(way_points == env_agent.goal):
                    continue
                agent_layer[way_points[0], way_points[1]] = agent_code + AGENT_GLOBAL_PATH_PLUS
            agent_layer[env_agent.pos[0], env_agent.pos[1]] = agent_code
            self.env.city_grid[env_agent.layer_id] = agent_layer
    
    def get_reward(self, obs_array, action):
        ''' Get the reward for the current step.
        :param np.array obs_array: the observation array
        :param int action: the action index
        :return: the reward
        '''
        # get the SAT reward/fail
        reward_eval = self.env.local_planner.eval_state_action(obs_array, action)
        if isinstance(reward_eval, tuple) and len(reward_eval) == 3:
            fail, sat_reward, _ = reward_eval
        else:
            fail, sat_reward = reward_eval
        if fail:
            return sat_reward
        moving_cost = self.action2cost(action)
        return (moving_cost + sat_reward)/self.path_length

    def action2cost(self, action):
        ''' Convert the action to cost.
        :param list action: the action list
        :return: the cost
        '''
        if action[0] == 1:
            # Slow
            return self.action_cost[0]
        elif action[4] == 1:
            # Normal
            return self.action_cost[1]
        elif action[8] == 1:
            # Fast
            return self.action_cost[2]
        else:
            # Stop
            return self.action_cost[3]
    
    
    def reset(self, return_info=False, seed=_RESET_SENTINEL, options=_RESET_SENTINEL):
        if seed is not _RESET_SENTINEL and seed is not None:
            self.seed(seed)
        logger.debug("***Reset RL Agent in Env***")
        max_attempts = 128
        for _ in range(max_attempts):
            self.t = 0
            for env_agent in self.env.agents:
                if getattr(env_agent, "cached_init_info", None) is not None:
                    env_agent.init(self.env.city_grid, init_info=env_agent.cached_init_info)
                else:
                    env_agent.init(self.env.city_grid)
                if env_agent.layer_id == self.agent_layer_id and getattr(env_agent, "cached_init_info", None) is None:
                    env_agent.reset_concepts(self.max_priority, self.reset_dist)
            if self._passes_reset_prefilter():
                break
        else:
            raise RuntimeError("Failed to sample a reset satisfying reset_prefilter after {} attempts.".format(max_attempts))
        logger.debug("Agent reset priority to %s/%s", self.agent.priority, self.max_priority)
        logger.debug("Agent reset concepts to %s", self.agent.concepts)
        self.path_length = len(self.agent.global_traj)*4
        self.normed_path_length = len(self.agent.global_traj)/(2*self.agent.region)
        self.env.local_planner.reset()
        self._redraw_all_agent_layers()
        if return_info:
            episode = self.save_episode()
        ob_dict = self.env.update(self.agent_layer_id)
        self.last_planner_actions = ob_dict.get("Planner_actions", {})
        self.last_agent_positions = ob_dict.get("Agent_positions", {})
        obs = self._flatten_obs(ob_dict)
        self.current_obs = obs
        self.last_dist = -1
        self.last_pos = None
        self.current_episode_reward = 0
        self.current_episode_length = 0
        if self.use_expert:
            self.expert_action = self.full_action2index(ob_dict["Expert_actions"][0])
            if return_info:
                return self.current_obs, episode
            expert_info = {}
            expert_info["Next_grounding"] = ob_dict["Ground_dic"][0]
            expert_info["Next_sg"] = ob_dict["Expert_sg"][0]
            expert_info["Planner_actions"] = ob_dict.get("Planner_actions", {})
            expert_info["Agent_positions"] = ob_dict.get("Agent_positions", {})
            return self.current_obs, expert_info
        else:
            return self.current_obs, {}
    
    def init(self):
        # init does not reset the agent
        logger.info("***Init RL Agent in Env***")
        self.t = 0
        self.path_length = len(self.agent.global_traj)*4
        self.normed_path_length = len(self.agent.global_traj)/(2*self.agent.region)
        self.env.local_planner.reset()
        ob_dict = self.env.update(self.agent_layer_id)
        self.last_planner_actions = ob_dict.get("Planner_actions", {})
        self.last_agent_positions = ob_dict.get("Agent_positions", {})
        if self.use_expert:
            self.expert_action = self.full_action2index(ob_dict["Expert_actions"][0])
        obs = self._flatten_obs(ob_dict)
        self.last_dist = -1
        self.last_pos = None
        self.current_obs = obs
        return self.current_obs
    
    def step(self, action):
        self.t += 1
        info = {}
        one_hot_action = torch.tensor(self.action_mapping[action], dtype=torch.float32)
        prev_distance = self._trajectory_distance_remaining()
        # move and get reward
        current_obs = self.env.move_rl_agent(one_hot_action, self.agent_layer_id)
        curr_distance = self._trajectory_distance_remaining()
        reached_goal = self.agent.reach_goal
        if self.reward_scheme == "safe_path_following":
            rew = self._get_spf_reward(current_obs, prev_distance, curr_distance, reached_goal)
        else:
            rew = self._get_reward(current_obs)
        info.update(current_obs)
        new_ob_dict = self.env.update(self.agent_layer_id)
        self.last_planner_actions = new_ob_dict.get("Planner_actions", {})
        self.last_agent_positions = new_ob_dict.get("Agent_positions", {})
        if self.use_expert:
            self.expert_action = self.full_action2index(new_ob_dict["Expert_actions"][0])
            info["Next_grounding"] = new_ob_dict["Ground_dic"][0]
            info["Next_sg"] = new_ob_dict["Expert_sg"][0]
            info["Planner_actions"] = new_ob_dict.get("Planner_actions", {})
            info["Agent_positions"] = new_ob_dict.get("Agent_positions", {})
        # ob_dict = self.env.update()
        self.current_episode_reward += rew
        self.current_episode_length += 1
        obs = self._flatten_obs(new_ob_dict)
        self.current_obs = obs
        
        # offset the index by 3 layers 0,1,2 are static in world matrix
        terminated = reached_goal
        truncated = False
        info["is_success"] = False

        if terminated:
            info['episode'] = {'r': self.current_episode_reward, 'l': self.current_episode_length}
            info["is_success"] = True
            logger.debug("will reset agent by success")
            self.reset()
        
        if self.t >= self.horizon: 
            truncated = True
            rew += self.overtime_cost
            info["overtime"] = True
            info['episode'] = {'r': self.current_episode_reward, 'l': self.current_episode_length}
            logger.debug("Reset agent by overtime")
            self.reset()
            
        if info["Fail"][0]: 
            terminated = True
            logger.debug("Reset agent by failing")
            info['episode'] = {'r': self.current_episode_reward, 'l': self.current_episode_length}
            self.reset()

        return self.current_obs, rew, terminated, truncated, info
    
    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
    
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
        episode = {
            "city_grid": city_grid,
            "agents": agents
        }
        return episode

    def check_success(self):
        return self.agent.reach_goal
    
    def seed(self, seed=None):
        if seed is not None:
            try:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
            except:
                TypeError("Seed must be an integer type!")
    
