from .car import Car
import torch
import numpy as np
from ..utils.find import interpolate_car_path
from ..planners import GPlanner_mapper
from ..core.config import *
import logging
# import cv2
# import numpy as np
# # vis quick tool
# vis = np.zeros((250, 250, 3))
# vis[self.movable_region] = [255, 0, 0]
# vis[self.goal[0], self.goal[1]] = [0, 255, 255]
# vis[self.start[0], self.start[1]] = [0, 255, 0]
# vis[self.midline_matrix] = [0, 0, 255]
# cv2.imwrite("test.png", vis)

logger = logging.getLogger(__name__)

TYPE_MAP = {v: k for k, v in LABEL_MAP.items()}

class Bus(Car):
    def __init__(self, size, id, world_state_matrix, global_planner, concepts, debug=False, region=240, sampling_config=None):
        super().__init__(size, id, world_state_matrix, global_planner, concepts, debug=debug, region=region, sampling_config=sampling_config)

    def init(self, world_state_matrix, init_info=None, debug=False):
        Traffic_STREET = TYPE_MAP['Traffic Street']
        CROSSING_STREET = TYPE_MAP['Overlap']
        self.movable_region = (world_state_matrix[STREET_ID] == Traffic_STREET) | (world_state_matrix[STREET_ID] == CROSSING_STREET)
        self.route2waypoints(BUS_ROUTES[self.concepts["no."]], max_step=1)
        self.start = self.global_traj[0].clone()
        self.pos = self.start.clone()
        self.goal = self.global_traj[-1].clone()
        self.midline_matrix = (world_state_matrix[STREET_ID] == Traffic_STREET+MID_LINE_CODE_PLUS)
        self.global_planner = GPlanner_mapper[self.global_planner_type](self.movable_region, self.midline_matrix, CAR_STREET_OFFSET)
        self.intersection_points = torch.cat([torch.cat(self.global_planner.start_lists, dim=0), torch.cat(self.global_planner.end_lists, dim=0)], dim=0)
        logger.debug("%s_%s initialization done!", self.type, self.id)

    def route2waypoints(self, route_list, max_step):
        road_nodes = np.loadtxt(ROAD_GRAPH_NODES)
        self.global_traj = []
        for i in range(0, len(route_list) - 1):
            start = tuple(road_nodes[route_list[i]].astype(int))
            goal = tuple(road_nodes[route_list[i+1]].astype(int))
            path = interpolate_car_path(self.movable_region, [start, goal], max_step)
            self.global_traj.extend(path[:-1])
        self.global_traj = torch.stack(self.global_traj, dim=0)

    def get_next_action(self, world_state_matrix, local_action_dist):
        # buses never reaches the goal
        return self.get_action(local_action_dist), world_state_matrix[self.layer_id]

    def move(self, action, ped_layer):
        curr_pos = torch.nonzero((ped_layer==TYPE_MAP[self.type]).float())[0]
        assert torch.all(self.pos == curr_pos)
        ped_layer[self.pos[0], self.pos[1]] = TYPE_MAP[self.type]+AGENT_GLOBAL_PATH_PLUS
        next_pos = self.pos.clone()
        # bus do not becomes walked grid
        next_pos += self.action_to_move.get(action.item(), torch.tensor((0, 0)))
        self.pos = next_pos.clone()
        # Update Agent Map
        ped_layer[self.start[0], self.start[1]] = TYPE_MAP[self.type] + AGENT_START_PLUS
        ped_layer[self.goal[0], self.goal[1]] = TYPE_MAP[self.type] + AGENT_GOAL_PLUS
        ped_layer[self.pos[0], self.pos[1]] = TYPE_MAP[self.type]
        return ped_layer
