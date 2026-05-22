import time
import torch
import logging
from .basic import Agent
from ..core.config import *
import torch.nn.functional as F
from torch.distributions import Categorical
from ..utils.find import find_nearest_building, find_building_mask
from ..planners import GPlanner_mapper
from ..utils.sample import sample_start_goal, sample_determine_start_goal

logger = logging.getLogger(__name__)

TYPE_MAP = {v: k for k, v in LABEL_MAP.items()}

class Pedestrian(Agent):
    def __init__(self, size, id, world_state_matrix, global_planner, concepts, init_info=None, debug=False, region=240):
        self.start_point_list = None
        self.goal_point_list = None
        self.global_planner = GPlanner_mapper[global_planner]
        super().__init__(size, id, world_state_matrix, concepts, init_info=init_info, debug=debug, region=region)
        # pedestrian use A*, which is just a function
        self.action_mapping = {
            0: "Left_Normal", 
            1: "Right_Normal", 
            2: "Up_Normal", 
            3: "Down_Normal", 
            4: "Stop"
            }
        self.action_to_move = {
            self.action_space[0].item(): torch.tensor((0, -1)),
            self.action_space[1].item(): torch.tensor((0, 1)),
            self.action_space[2].item(): torch.tensor((-1, 0)),
            self.action_space[3].item(): torch.tensor((1, 0))
        }
        self.move_to_action = {
            self.action_to_move[k]: k for k in self.action_to_move
        }
    
    def init(self, world_state_matrix, init_info=None, debug=False):
        WALKING_STREET = TYPE_MAP['Walking Street']
        CROSSING_STREET = TYPE_MAP['Overlap']
        if init_info is not None:
            self.init_from_dict(init_info)
            _ = self.get_start(world_state_matrix)
            _ = self.get_goal(world_state_matrix, self.start)
        else:
            if debug:
                self.start, self.goal = sample_determine_start_goal(self.type, self.id)
                self.pos = self.start.clone()
            else:
                self.start, self.goal = self._sample_start_and_goal(world_state_matrix)
                self.pos = self.start.clone()
        # specify the occupacy map
        self.movable_region = (world_state_matrix[STREET_ID] == WALKING_STREET) | (world_state_matrix[STREET_ID] == CROSSING_STREET)
        # get global traj on the occupacy map
        self.global_traj = self.global_planner(self.movable_region, self.start, self.goal)
        self.reach_goal = False
        self.last_move_dir = None
        logger.info("{}_{} initialization done!".format(self.type, self.id))

    def _sample_start_and_goal(self, world_state_matrix, max_attempts=64):
        for _ in range(max_attempts):
            start = torch.tensor(self.get_start(world_state_matrix))
            goal = self.get_goal(world_state_matrix, start)
            if goal is not None:
                return start, torch.tensor(goal)
        raise RuntimeError("Failed to sample a valid pedestrian start/goal pair after {} attempts.".format(max_attempts))

    def _pedestrian_candidate_mask(self, world_state_matrix, building_types):
        building = [TYPE_MAP[b] for b in building_types]
        desired_locations = sample_start_goal(
            world_state_matrix,
            TYPE_MAP['Walking Street'],
            building,
            kernel_size=PED_GOAL_START_INCLUDE_KERNEL,
        )
        desired_locations[self.region:, :] = False
        desired_locations[:, self.region:] = False
        return desired_locations

    def get_start(self, world_state_matrix):
        # Prefer house/office-adjacent starts, but relax to any building-adjacent
        # walking street when the small map makes the preferred set empty.
        desired_locations = self._pedestrian_candidate_mask(world_state_matrix, PEDES_GOAL_START)
        if torch.nonzero(desired_locations).numel() == 0:
            desired_locations = self._pedestrian_candidate_mask(world_state_matrix, BUILDING_TYPES)
        
        self.start_point_list = torch.nonzero(desired_locations).tolist()
        if len(self.start_point_list) == 0:
            raise RuntimeError("No valid pedestrian start positions found for the current map/region.")
        random_index = torch.randint(0, len(self.start_point_list), (1,)).item()
        
        # Fetch the corresponding location
        start_point = self.start_point_list[random_index]

        # Return the indices of the desired locations
        return start_point
    
    def get_goal(self, world_state_matrix, start_point):
        # Prefer house/office-adjacent goals, but relax to any building-adjacent
        # walking street when the preferred set is too small on compact maps.
        self.desired_locations = self._pedestrian_candidate_mask(world_state_matrix, PEDES_GOAL_START)
        if torch.nonzero(self.desired_locations).numel() == 0:
            self.desired_locations = self._pedestrian_candidate_mask(world_state_matrix, BUILDING_TYPES)
        desired_locations = self.desired_locations.detach().clone()
        
        # Determine the nearest building to the start point
        nearest_building = find_nearest_building(world_state_matrix, start_point)
        start_block = world_state_matrix[BLOCK_ID][nearest_building[0], nearest_building[1]]

        # Get the mask for the building containing the nearest_building position
        # building_mask = find_building_mask(world_state_matrix, nearest_building)
        building_mask = world_state_matrix[BLOCK_ID] == start_block
        
        # Create a mask to exclude areas around the building. We'll dilate the building mask.
        exclusion_radius = PED_GOAL_START_EXCLUDE_KERNEL  # Excludes surrounding 5 grids around the block
        expanded_mask = F.max_pool2d(building_mask[None, None].float(), exclusion_radius, stride=1, padding=(exclusion_radius - 1) // 2) > 0
        
        desired_locations[expanded_mask[0, 0]] = False
        desired_locations[self.region:, :] = False
        desired_locations[:, self.region:] = False
        # Return the indices of the desired locations
        goal_point_list = torch.nonzero(desired_locations).tolist()
        if len(goal_point_list) == 0:
            # On the 2x2 map the exclusion kernel can wipe out every preferred
            # candidate, so relax by allowing any building-adjacent walking street
            # outside the exact start cell.
            relaxed_locations = self._pedestrian_candidate_mask(world_state_matrix, BUILDING_TYPES)
            relaxed_locations[start_point[0], start_point[1]] = False
            goal_point_list = torch.nonzero(relaxed_locations).tolist()
        if len(goal_point_list) == 0:
            return None
        random_index = torch.randint(0, len(goal_point_list), (1,)).item()
        
        # Fetch the corresponding location
        goal_point = goal_point_list[random_index]

        # Return the indices of the desired locations
        return goal_point

    def get_next_action(self, world_state_matrix, local_action_dist):
        # for now, just reckless take the global traj
        # reached goal
        if not self.reach_goal:
            if torch.all(self.pos == self.goal):
                if not self.debug:
                    self.reach_goal = True
                    self.reach_goal_buffer += REACH_GOAL_WAITING
                #     logger.info("{}_{} reached goal! Will change goal in the next {} step!".format(self.type, self.id, self.reach_goal_buffer))
                # else:
                #     logger.info("{}_{} reached goal! In Debug, it will stop".format(self.type, self.id))
                return self.action_space[-1], world_state_matrix[self.layer_id]
            else:
                return self.get_action(local_action_dist), world_state_matrix[self.layer_id]
        else:
            if self.reach_goal_buffer > 0:
                self.reach_goal_buffer -= 1
                # logger.info("{}_{} reached goal! Will change goal in the next {} step!".format(self.type, self.id, self.reach_goal_buffer))
                return self.action_space[-1], world_state_matrix[self.layer_id]
            # logger.info("Generating new goal and gloabl plans for {}_{}...".format(self.type, self.id))
            self.start = self.goal.clone()
            desired_locations = self.desired_locations.detach().clone()

            # Determine the nearest building to the start point
            nearest_building = find_nearest_building(world_state_matrix, self.start)
            start_block = world_state_matrix[BLOCK_ID][nearest_building[0], nearest_building[1]]

            # Get the mask for the building containing the nearest_building position
            building_mask = world_state_matrix[BLOCK_ID] == start_block
            
            # Create a mask to exclude areas around the building. We'll dilate the building mask.
            exclusion_radius = PED_GOAL_START_EXCLUDE_KERNEL  # Excludes surrounding 5 grids around the block
            expanded_mask = F.max_pool2d(building_mask[None, None].float(), exclusion_radius, stride=1, padding=(exclusion_radius - 1) // 2) > 0
            
            desired_locations[expanded_mask[0, 0]] = False
            desired_locations[self.region:, :] = False
            desired_locations[:, self.region:] = False

            # Return the indices of the desired locations
            goal_point_list = torch.nonzero(desired_locations).tolist()
            if len(goal_point_list) == 0:
                self.start, self.goal = self._sample_start_and_goal(world_state_matrix)
                self.pos = self.start.clone()
                self.global_traj = self.global_planner(self.movable_region, self.start, self.goal)
                self.reach_goal = False
                self.last_move_dir = None
                world_state_matrix[self.layer_id] *= 0
                world_state_matrix[self.layer_id][self.goal[0], self.goal[1]] = TYPE_MAP[self.type] + AGENT_GOAL_PLUS
                for way_points in self.global_traj[1:-1]:
                    world_state_matrix[self.layer_id][way_points[0], way_points[1]] \
                        = TYPE_MAP[self.type] + AGENT_GLOBAL_PATH_PLUS
                world_state_matrix[self.layer_id][self.start[0], self.start[1]] = TYPE_MAP[self.type]
                return self.get_action(local_action_dist), world_state_matrix[self.layer_id]
            random_index = torch.randint(0, len(goal_point_list), (1,)).item()
            
            # Fetch the corresponding location
            self.goal = torch.tensor(goal_point_list[random_index])
            self.global_traj = self.global_planner(self.movable_region, self.start, self.goal)
            # logger.info("Generating new goal and gloabl plans for {}_{} done!".format(self.type, self.id))
            self.reach_goal = False
            self.last_move_dir = None
            # delete past traj
            world_state_matrix[self.layer_id] *= 0
            world_state_matrix[self.layer_id][self.goal[0], self.goal[1]] = TYPE_MAP[self.type] + AGENT_GOAL_PLUS
            for way_points in self.global_traj[1:-1]:
                world_state_matrix[self.layer_id][way_points[0], way_points[1]] \
                    = TYPE_MAP[self.type] + AGENT_GLOBAL_PATH_PLUS
            world_state_matrix[self.layer_id][self.start[0], self.start[1]] = TYPE_MAP[self.type]

            return self.get_action(local_action_dist), world_state_matrix[self.layer_id]
