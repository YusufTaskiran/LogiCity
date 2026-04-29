import numpy as np
import torch

from .city import City
import gym
from .config import *
from ..utils.vis import visualize_city
from ..utils.gym_wrapper import GymCityWrapper

class CityEnv(City):
    def __init__(self, grid_size, local_planner, logic_engine_file, rl_agent, use_multi=False):
        super().__init__(grid_size, local_planner, logic_engine_file, use_multi=use_multi)
        self.rl_agent = rl_agent
        self.logic_grounding_shape, self.pred_grounding_index = self.local_planner.logic_grounding_shape(self.rl_agent["fov_entities"])

    def move_rl_agent(self, action, idx):
        current_obs = {}
        current_obs["Fail"] = []
        current_obs["Agent_actions"] = []
        current_obs["Reward"] = []
        
        fail, reward = self.local_planner.eval(action)
        current_obs["Fail"].append(fail)
        current_obs["Reward"].append(reward)
        new_matrix = torch.zeros_like(self.city_grid)
        
        for agent in self.agents:
            # re-initialized agents may update city matrix as well
            # local reasoning-based action distribution
            # global trajectory-based action or sampling from local action distribution
            if agent.layer_id == idx: 
                current_obs["Agent_actions"].append(action)
                local_action, new_matrix[agent.layer_id] = agent.get_next_action(self.city_grid, action)
            else: 
                continue

            if agent.reach_goal:
                continue

            next_layer = agent.move(local_action, new_matrix[agent.layer_id])
            # print(torch.nonzero(next_layer), np.unique(next_layer), torch.nonzero((next_layer==8.0).float())[0])
            new_matrix[agent.layer_id] = next_layer
        # Update city grid after all the agents make decisions
        self.city_grid[idx] = new_matrix[idx]
        current_obs["World"] = self.city_grid.clone()
        return current_obs

    def update(self, idx):
        current_obs = {}
        # state at time t
        current_obs["World_state"] = []
        current_obs["Expert_actions"] = []
        current_obs["Expert_sg"] = []
        current_obs["Ground_dic"] = []
        current_obs["Shield_context"] = []

        new_matrix = torch.zeros_like(self.city_grid)
        current_world = self.city_grid.clone()
        # first do local planning based on city rules
        agent_action_dist = self.local_planner.plan(current_world, self.intersection_matrix, self.agents, \
                                                    self.layer_id2agent_list_id, use_multiprocessing=self.use_multi, rl_agent=idx)
        # Then do global action taking acording to the local planning results
        # input((action_idx, idx))
        
        for agent in self.agents:
            # re-initialized agents may update city matrix as well
            agent_name = "{}_{}".format(agent.type, agent.layer_id)
            # local reasoning-based action distribution
            # global trajectory-based action or sampling from local action distribution
            if agent.layer_id == idx: 
                current_obs["World_state"].append(agent_action_dist["{}_grounding".format(agent_name)])
                new_matrix[agent.layer_id] = self.city_grid[agent.layer_id].clone()
                # expert will provide the action and scene graph and groundings
                if "{}_grounding_dic".format(agent_name) in agent_action_dist:
                    current_obs["Ground_dic"].append(agent_action_dist["{}_grounding_dic".format(agent_name)])
                if "{}_shield_context".format(agent_name) in agent_action_dist:
                    current_obs["Shield_context"].append(agent_action_dist["{}_shield_context".format(agent_name)])
                if "{}_scene_graph".format(agent_name) in agent_action_dist:
                    current_obs["Expert_sg"].append(agent_action_dist["{}_scene_graph".format(agent_name)])
                if "{}_action".format(agent_name) in agent_action_dist:
                    current_obs["Expert_actions"].append(agent_action_dist["{}_action".format(agent_name)].clone())
                continue
            else: 
                local_action_dist = agent_action_dist[agent_name]
                local_action, new_matrix[agent.layer_id] = agent.get_next_action(self.city_grid, local_action_dist)

            if agent.reach_goal:
                continue

            next_layer = agent.move(local_action, new_matrix[agent.layer_id])
            # print(torch.nonzero(next_layer), np.unique(next_layer), torch.nonzero((next_layer==8.0).float())[0])
            new_matrix[agent.layer_id] = next_layer
        # Update city grid after all the agents make decisions
        self.city_grid[BASIC_LAYER:] = new_matrix[BASIC_LAYER:]
        return current_obs
