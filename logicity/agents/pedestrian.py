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
    def __init__(self, size, id, world_state_matrix, global_planner, concepts, init_info=None, debug=False, region=240, sampling_config=None):
        self.start_point_list = None
        self.goal_point_list = None
        self.global_planner = GPlanner_mapper[global_planner]
        super().__init__(size, id, world_state_matrix, concepts, init_info=init_info, debug=debug, region=region, sampling_config=sampling_config)
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
        self.movable_region = (world_state_matrix[STREET_ID] == WALKING_STREET) | (world_state_matrix[STREET_ID] == CROSSING_STREET)
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
        # get global traj on the occupacy map
        self.global_traj = self.global_planner(self.movable_region, self.start, self.goal)
        self.reach_goal = False
        self.last_move_dir = None
        logger.debug("%s_%s initialization done!", self.type, self.id)

    def _sample_start_and_goal(self, world_state_matrix, max_attempts=64):
        for _ in range(max_attempts):
            start = torch.tensor(self.get_start(world_state_matrix))
            goal = self.get_goal(world_state_matrix, start)
            if goal is not None:
                return start, torch.tensor(goal)
        raise RuntimeError("Failed to sample a valid pedestrian start/goal pair after {} attempts.".format(max_attempts))

    def _pedestrian_sampling_cfg(self):
        if not isinstance(self.sampling_config, dict):
            return {}
        return self.sampling_config.get("pedestrian", {}) or {}

    def _pedestrian_center_constraint_active(self):
        ped_cfg = self._pedestrian_sampling_cfg()
        if ped_cfg.get("goal_constraint_mode") != "cross_center_intersection":
            return False
        constrained_ids = ped_cfg.get("constrained_ids", None)
        if constrained_ids is None:
            return True
        return int(self.id) in {int(v) for v in constrained_ids}

    def _center_intersection_bbox(self, grid_shape):
        ped_cfg = self._pedestrian_sampling_cfg()
        bbox_pad = int(ped_cfg.get("intersection_bbox_pad", TRAFFIC_STREET_WID + WALKING_STREET_WID))
        center_x = grid_shape[0] // 2
        center_y = grid_shape[1] // 2
        x_min = max(0, center_x - bbox_pad)
        x_max = min(grid_shape[0], center_x + bbox_pad + 1)
        y_min = max(0, center_y - bbox_pad)
        y_max = min(grid_shape[1], center_y + bbox_pad + 1)
        return x_min, x_max, y_min, y_max

    def _center_crossing_candidate_mask(self, world_state_matrix):
        x_min, x_max, y_min, y_max = self._center_intersection_bbox(self.movable_region.shape)
        mask = torch.zeros_like(world_state_matrix[STREET_ID], dtype=torch.bool)
        mask[x_min:x_max, y_min:y_max] = True
        walking_only = world_state_matrix[STREET_ID] == TYPE_MAP['Walking Street']
        desired_locations = walking_only & mask
        if torch.nonzero(desired_locations).numel() == 0:
            desired_locations = self.movable_region & mask
        return self.apply_region_mask(desired_locations)

    def _start_distance_offset_for_id(self):
        ped_cfg = self._pedestrian_sampling_cfg()
        offset_map = ped_cfg.get("start_distance_offset_by_id", {}) or {}
        return int(offset_map.get(int(self.id), 0))

    def _center_distance_metric(self, point, grid_shape):
        center_x = grid_shape[0] // 2
        center_y = grid_shape[1] // 2
        point_tensor = point if torch.is_tensor(point) else torch.tensor(point)
        dx = abs(int(point_tensor[0].item()) - center_x)
        dy = abs(int(point_tensor[1].item()) - center_y)
        return max(dx, dy)

    def _filter_start_candidates_by_offset(self, start_point_list):
        if len(start_point_list) == 0:
            return start_point_list

        desired_offset = self._start_distance_offset_for_id()
        if desired_offset <= 0:
            return start_point_list

        ped_cfg = self._pedestrian_sampling_cfg()
        tolerance = int(ped_cfg.get("start_distance_tolerance", 2))
        grid_shape = self.movable_region.shape

        exact_band = []
        relaxed_band = []
        for start_point in start_point_list:
            dist = self._center_distance_metric(start_point, grid_shape)
            if desired_offset <= dist <= desired_offset + tolerance:
                exact_band.append(start_point)
            if dist >= desired_offset:
                relaxed_band.append(start_point)

        if len(exact_band) > 0:
            return exact_band
        if len(relaxed_band) > 0:
            return relaxed_band
        return start_point_list

    def _center_side_label(self, point, grid_shape):
        center_x = grid_shape[0] // 2
        center_y = grid_shape[1] // 2
        point_tensor = point if torch.is_tensor(point) else torch.tensor(point)
        dx = int(point_tensor[0].item()) - center_x
        dy = int(point_tensor[1].item()) - center_y

        if abs(dx) >= abs(dy):
            return "south" if dx > 0 else "north"
        return "east" if dy > 0 else "west"

    def _goal_crosses_center_filter(self, start_point, goal_point_list):
        if len(goal_point_list) == 0 or not self._pedestrian_center_constraint_active():
            return goal_point_list

        ped_cfg = self._pedestrian_sampling_cfg()
        min_goal_manhattan_distance = int(ped_cfg.get("min_goal_manhattan_distance", 6))
        start_tensor = start_point if torch.is_tensor(start_point) else torch.tensor(start_point)
        require_different_side = bool(ped_cfg.get("require_goal_different_center_side", True))
        start_side = self._center_side_label(start_tensor, self.movable_region.shape) if require_different_side else None
        x_min, x_max, y_min, y_max = self._center_intersection_bbox(self.movable_region.shape)
        max_goals_to_check = int(ped_cfg.get("max_goal_candidates_to_check", 24))
        accepted_goal_limit = int(ped_cfg.get("accepted_goal_pool_size", 8))
        accepted = []
        checked = 0

        for goal_point in goal_point_list:
            goal_tensor = torch.tensor(goal_point)
            if torch.abs(goal_tensor - start_tensor).sum().item() < min_goal_manhattan_distance:
                continue
            if require_different_side:
                goal_side = self._center_side_label(goal_tensor, self.movable_region.shape)
                if goal_side == start_side:
                    continue
            try:
                route = self.global_planner(self.movable_region, start_tensor, goal_tensor)
            except Exception:
                continue
            checked += 1
            route_x = route[:, 0]
            route_y = route[:, 1]
            crosses_center = bool(
                ((route_x >= x_min) & (route_x < x_max) & (route_y >= y_min) & (route_y < y_max)).any().item()
            )
            if not crosses_center:
                if max_goals_to_check > 0 and checked >= max_goals_to_check:
                    break
                continue
            accepted.append(goal_point)
            if len(accepted) >= accepted_goal_limit:
                break
            if max_goals_to_check > 0 and checked >= max_goals_to_check:
                break

        return accepted if len(accepted) > 0 else goal_point_list

    def _pedestrian_candidate_mask(self, world_state_matrix, building_types):
        building = [TYPE_MAP[b] for b in building_types]
        desired_locations = sample_start_goal(
            world_state_matrix,
            TYPE_MAP['Walking Street'],
            building,
            kernel_size=PED_GOAL_START_INCLUDE_KERNEL,
        )
        return self.apply_region_mask(desired_locations)

    def get_start(self, world_state_matrix):
        if self._pedestrian_center_constraint_active():
            desired_locations = self._center_crossing_candidate_mask(world_state_matrix)
            self.start_point_list = torch.nonzero(desired_locations).tolist()
            self.start_point_list = self._filter_start_candidates_by_offset(self.start_point_list)
            if len(self.start_point_list) == 0:
                raise RuntimeError("No valid center-constrained pedestrian start positions found.")
            random_index = torch.randint(0, len(self.start_point_list), (1,)).item()
            return self.start_point_list[random_index]

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
        if self._pedestrian_center_constraint_active():
            desired_locations = self._pedestrian_candidate_mask(world_state_matrix, PEDES_GOAL_START)
            if torch.nonzero(desired_locations).numel() == 0:
                desired_locations = self._pedestrian_candidate_mask(world_state_matrix, BUILDING_TYPES)
            self.desired_locations = desired_locations.detach().clone()
            desired_locations[start_point[0], start_point[1]] = False
            goal_point_list = torch.nonzero(desired_locations).tolist()
            goal_point_list = self._goal_crosses_center_filter(start_point, goal_point_list)
            if len(goal_point_list) == 0:
                return None
            random_index = torch.randint(0, len(goal_point_list), (1,)).item()
            return goal_point_list[random_index]

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
        desired_locations = self.apply_region_mask(desired_locations)
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
                self.reach_goal = True
                self.reach_goal_buffer += REACH_GOAL_WAITING
                # Goal completion must still latch in debug runs; otherwise debug evaluation can keep
                # already-finished agents alive until horizon.
                return self.action_space[-1], world_state_matrix[self.layer_id]
            else:
                return self.get_action(local_action_dist), world_state_matrix[self.layer_id]
        else:
            if self.reach_goal_buffer > 0:
                self.reach_goal_buffer -= 1
            return self.action_space[-1], world_state_matrix[self.layer_id]
