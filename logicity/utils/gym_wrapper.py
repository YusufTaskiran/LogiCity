import os
import gym
import torch
import numpy as np
import pickle as pkl
from gym.spaces import Box, Dict
import torch.nn.functional as F
from ..core.config import *
from ..shields import PredicateSensorModel

import logging
logger = logging.getLogger(__name__)

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
        self.type2label = {v: k for k, v in LABEL_MAP.items()}
        self.scale = [25, 7, 3.5, 8.3]
        self.mini_scale = [0, 0, -1, 0]
        self.t = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.require_route_intersection = env.rl_agent.get("require_route_intersection", False)
        self.reset_attempts = env.rl_agent.get("reset_attempts", 20)
        self.reset_all_agents = env.rl_agent.get("reset_all_agents", True)
        self.randomize_all_car_priorities = env.rl_agent.get("randomize_all_car_priorities", False)
        self.shield_config = env.rl_agent.get("shield")
        self.observation_uncertainty_cfg = env.rl_agent.get("observation_uncertainty")
        self.observation_sensor_model = PredicateSensorModel(self.observation_uncertainty_cfg)
        self.observation_action_profile = (
            self.observation_uncertainty_cfg.get("action_profile", "normal")
            if self.observation_uncertainty_cfg
            else "normal"
        )
        self.current_grounding_dic = None
        self.training_episode_data_path = env.rl_agent.get("training_episode_data")
        self.training_episode_data = None
        self.training_episode_keys = []
        if self.training_episode_data_path:
            if not os.path.isfile(self.training_episode_data_path):
                raise FileNotFoundError(f"Training episode data not found: {self.training_episode_data_path}")
            with open(self.training_episode_data_path, "rb") as f:
                self.training_episode_data = pkl.load(f)
            self.training_episode_keys = list(self.training_episode_data.keys())
            if len(self.training_episode_keys) == 0:
                raise ValueError(f"Training episode data is empty: {self.training_episode_data_path}")
            logger.info(
                "Loaded %s cached training episodes from %s",
                len(self.training_episode_keys),
                self.training_episode_data_path,
            )

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

    def full_action2index(self, action):
        return self._macro_action_index(action)
    
    def _apply_observation_uncertainty(self, grounding: np.ndarray) -> np.ndarray:
        grounding = np.asarray(grounding, dtype=np.float32).copy()
        if not self.observation_sensor_model.enabled:
            return grounding
        for pred_name, (start, end) in self.pred_grounding_index.items():
            if not self.observation_sensor_model.is_uncertain(pred_name):
                continue
            for idx in range(start, end):
                truth_value = bool(grounding[idx] > 0.5)
                grounding[idx] = self.observation_sensor_model.sense_probability(
                    pred_name,
                    truth_value,
                    action_name=self.observation_action_profile,
                )
        return grounding

    def _flatten_obs(self, obs_dict):
        world_state = self._apply_observation_uncertainty(obs_dict["World_state"][0])
        if self.cat_length:
            return np.concatenate([world_state, [self.normed_path_length]], axis=0, dtype=np.float32)
        else:
            return world_state
                
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
    
    def get_reward(self, obs_array, action):
        ''' Get the reward for the current step.
        :param np.array obs_array: the observation array
        :param int action: the action index
        :return: the reward
        '''
        # get the SAT reward/fail
        fail, sat_reward = self.env.local_planner.eval_state_action(obs_array, action)
        if fail:
            return sat_reward
        moving_cost = self.action2cost(action)
        return (moving_cost + sat_reward)/self.path_length

    def action2cost(self, action):
        ''' Convert the action to cost.
        :param list action: the action list
        :return: the cost
        '''
        action_idx = self._macro_action_index(action)
        return self.action_cost[action_idx]

    def _path_to_set(self, path):
        return {tuple(int(v) for v in point.tolist()) for point in path}

    def _routes_intersect(self):
        car_path = self._path_to_set(self.agent.global_traj)
        other_agents = [agent for agent in self.env.agents if agent.layer_id != self.agent.layer_id]
        if not other_agents:
            return True
        for other_agent in other_agents:
            other_path = self._path_to_set(other_agent.global_traj)
            if car_path.intersection(other_path):
                return True
        return False

    def _redraw_agent_layer(self, agent):
        agent_code = self.type2label[agent.type]
        agent_layer = torch.zeros((self.env.grid_size[0], self.env.grid_size[1]))
        agent_layer[agent.start[0], agent.start[1]] = agent_code
        agent_layer[agent.goal[0], agent.goal[1]] = agent_code + AGENT_GOAL_PLUS
        for way_points in agent.global_traj[1:-1]:
            if torch.all(way_points == agent.start) or torch.all(way_points == agent.goal):
                continue
            agent_layer[way_points[0], way_points[1]] = agent_code + AGENT_GLOBAL_PATH_PLUS
        self.env.city_grid[agent.layer_id] = agent_layer

    def _sample_episode_layout(self):
        if self.reset_all_agents:
            for scene_agent in self.env.agents:
                scene_agent.init(self.env.city_grid)
        else:
            self.agent.init(self.env.city_grid)
        if self.randomize_all_car_priorities:
            for scene_agent in self.env.agents:
                if hasattr(scene_agent, "reset_concepts") and scene_agent.type == "Car":
                    scene_agent.reset_concepts(self.max_priority, self.reset_dist)
        else:
            if hasattr(self.agent, "reset_concepts"):
                self.agent.reset_concepts(self.max_priority, self.reset_dist)
        for scene_agent in self.env.agents:
            self._redraw_agent_layer(scene_agent)

    def _load_cached_episode_layout(self):
        episode_key = np.random.choice(self.training_episode_keys)
        episode_cache = self.training_episode_data[episode_key]
        self.env.city_grid = episode_cache["city_grid"].clone()
        for scene_agent in self.env.agents:
            agent_name = f"{scene_agent.type}_{scene_agent.id}"
            if agent_name not in episode_cache["agents"]:
                raise KeyError(f"Agent {agent_name} missing from cached episode {episode_key}")
            scene_agent.init(self.env.city_grid, init_info=episode_cache["agents"][agent_name])
            scene_agent.reach_goal = False
            scene_agent.reach_goal_buffer = 0
            self._redraw_agent_layer(scene_agent)
        logger.info("Reset from cached training episode %s.", episode_key)

    
    def reset(self, return_info=False):
        logger.info("***Reset RL Agent in Env***")
        self.t = 0
        if self.training_episode_data is not None:
            self._load_cached_episode_layout()
        else:
            route_ok = False
            for attempt in range(1, self.reset_attempts + 1):
                self._sample_episode_layout()
                route_ok = (not self.require_route_intersection) or self._routes_intersect()
                if route_ok:
                    logger.info("Accepted reset layout after %s attempt(s).", attempt)
                    break
            if not route_ok:
                logger.info("Failed to satisfy route-intersection filter after %s attempt(s); using last sampled layout.", self.reset_attempts)
        logger.info("Agent reset priority to {}/{}".format(self.agent.priority, self.max_priority))
        logger.info("Agent reset concepts to {}".format(self.agent.concepts))
        self.path_length = len(self.agent.global_traj)*4
        self.normed_path_length = len(self.agent.global_traj)/(2*self.agent.region)
        self.env.local_planner.reset()
        if return_info:
            episode = self.save_episode()
        ob_dict = self.env.update(self.agent_layer_id)
        obs = self._flatten_obs(ob_dict)
        self.current_obs = obs
        self.last_dist = -1
        self.last_pos = None
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_grounding_dic = ob_dict["Ground_dic"][0] if len(ob_dict["Ground_dic"]) > 0 else None
        if self.use_expert:
            self.expert_action = self.full_action2index(ob_dict["Expert_actions"][0])
            if return_info:
                return self.current_obs, episode
            expert_info = {}
            expert_info["Next_grounding"] = ob_dict["Ground_dic"][0]
            expert_info["Next_sg"] = ob_dict["Expert_sg"][0]
            return self.current_obs, expert_info
        else:
            return self.current_obs
    
    def init(self):
        # init does not reset the agent
        logger.info("***Init RL Agent in Env***")
        self.t = 0
        self.path_length = len(self.agent.global_traj)*4
        self.normed_path_length = len(self.agent.global_traj)/(2*self.agent.region)
        self.env.local_planner.reset()
        ob_dict = self.env.update(self.agent_layer_id)
        if self.use_expert:
            self.expert_action = self.full_action2index(ob_dict["Expert_actions"][0])
        obs = self._flatten_obs(ob_dict)
        self.last_dist = -1
        self.last_pos = None
        self.current_obs = obs
        self.current_grounding_dic = ob_dict["Ground_dic"][0] if len(ob_dict["Ground_dic"]) > 0 else None
        return self.current_obs

    def step(self, action):
        self.t += 1
        info = {}
        one_hot_action = torch.tensor(self.action_mapping[action], dtype=torch.float32)
        # move and get reward
        current_obs = self.env.move_rl_agent(one_hot_action, self.agent_layer_id)
        rew = self._get_reward(current_obs)
        info.update(current_obs)
        new_ob_dict = self.env.update(self.agent_layer_id)
        if self.use_expert:
            self.expert_action = self.full_action2index(new_ob_dict["Expert_actions"][0])
            info["Next_grounding"] = new_ob_dict["Ground_dic"][0]
            info["Next_sg"] = new_ob_dict["Expert_sg"][0]
        # ob_dict = self.env.update()
        self.current_episode_reward += rew
        self.current_episode_length += 1
        obs = self._flatten_obs(new_ob_dict)
        self.current_obs = obs
        self.current_grounding_dic = new_ob_dict["Ground_dic"][0] if len(new_ob_dict["Ground_dic"]) > 0 else None
        
        # offset the index by 3 layers 0,1,2 are static in world matrix
        done = self.agent.reach_goal
        info["is_success"] = False

        if done:
            info['episode'] = {'r': self.current_episode_reward, 'l': self.current_episode_length}
            info["is_success"] = True
            logger.info("Episode ended by success.")
        
        if self.t >= self.horizon: 
            done = True
            rew += self.overtime_cost
            info["overtime"] = True
            info['episode'] = {'r': self.current_episode_reward, 'l': self.current_episode_length}
            logger.info("Episode ended by overtime.")
            
        if info["Fail"][0]: 
            done = True
            info['episode'] = {'r': self.current_episode_reward, 'l': self.current_episode_length}
            logger.info("Episode ended by failure.")

        return self.current_obs, rew, done, info
    
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
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")
    
