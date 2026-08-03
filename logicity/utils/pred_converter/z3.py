import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from ...core.config import *
import logging

logger = logging.getLogger(__name__)

TYPE_MAP = {v: k for k, v in LABEL_MAP.items()}

def IsCar(world_matrix, intersect_matrix, agents, entity):
    if "PH" in entity:
        return 0
    if "Car" in entity:
        return 1
    else:
        return 0
    
def IsPed(world_matrix, intersect_matrix, agents, entity):
    if "PH" in entity:
        return 0
    if "Pedestrian" in entity:
        return 1
    else:
        return 0

def IsAmb(world_matrix, intersect_matrix, agents, entity):
    if "Pedestrian" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    else:
        assert "ego_{}".format(layer_id) in agents.keys()
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    if "ambulance" in agent_concept:
        return 1
    else:
        return 0

def IsBus(world_matrix, intersect_matrix, agents, entity):
    if "Pedestrian" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    else:
        assert "ego_{}".format(layer_id) in agents.keys()
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    if "bus" in agent_concept:
        return 1
    else:
        return 0
    
def IsTiro(world_matrix, intersect_matrix, agents, entity):
    assert "Agent" in entity
    if "Pedestrian" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    else:
        assert "ego_{}".format(layer_id) in agents.keys()
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    if "tiro" in agent_concept:
        return 1
    else:
        return 0
    
def IsPolice(world_matrix, intersect_matrix, agents, entity):
    if "Pedestrian" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    else:
        assert "ego_{}".format(layer_id) in agents.keys()
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    if "police" in agent_concept:
        return 1
    else:
        return 0

def IsTiro(world_matrix, intersect_matrix, agents, entity):
    if "Pedestrian" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    else:
        assert "ego_{}".format(layer_id) in agents.keys()
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    if "tiro" in agent_concept:
        return 1
    else:
        return 0

def IsReckless(world_matrix, intersect_matrix, agents, entity):
    if "Pedestrian" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    else:
        assert "ego_{}".format(layer_id) in agents.keys()
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    if "reckless" in agent_concept:
        return 1
    else:
        return 0
    
def IsOld(world_matrix, intersect_matrix, agents, entity):
    if "Car" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    else:
        assert "ego_{}".format(layer_id) in agents.keys()
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    if "old" in agent_concept:
        return 1
    else:
        return 0
    
def IsYoung(world_matrix, intersect_matrix, agents, entity):
    if "Car" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    else:
        assert "ego_{}".format(layer_id) in agents.keys()
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    if "young" in agent_concept:
        return 1
    else:
        return 0
    
def IsAtInter(world_matrix, intersect_matrix, agents, entity1):
    if "PH" in entity1:
        return 0

    _, agent_type, layer_id = entity1.split("_")
    layer_id = int(layer_id)
    agent_layer = world_matrix[layer_id]
    agent_position = (agent_layer == TYPE_MAP[agent_type]).nonzero()[0]
    # For cars, "at intersection" should capture the route boundary of an
    # intersection and the immediate approach zone for multi-cell actions.
    # Fast/normal actions can otherwise jump over the thin boundary mask and
    # land directly inside the intersection block. We therefore treat a car as
    # "at" the intersection when it is currently outside the block but will
    # enter that block within the next few route cells.
    if agent_type == "Car":
        if intersect_matrix[2, agent_position[0], agent_position[1]]:
            return 0
        agent_key = str(layer_id) if str(layer_id) in agents else "ego_{}".format(layer_id)
        assert agent_key in agents, "Agent {} not found in predicate context".format(layer_id)
        agent = agents[agent_key]
        traj = getattr(agent, "global_traj", None)
        if traj is None or len(traj) == 0:
            return 0

        matches = torch.all(traj == agent_position, dim=1).nonzero(as_tuple=False)
        if matches.numel() == 0:
            return 0

        current_idx = int(matches[0].item())
        if intersect_matrix[0, agent_position[0], agent_position[1]]:
            return 1

        h, w = intersect_matrix.shape[1], intersect_matrix.shape[2]
        # A fast car can advance three route cells in one action. We therefore
        # need to look one cell beyond that final landing cell so the predicate
        # becomes active before a fast jump can place the car onto the last
        # crosswalk-adjacent waiting cell.
        lookahead_limit = min(len(traj), current_idx + 5)
        for idx in range(current_idx + 1, lookahead_limit):
            point = traj[idx]
            if point[0] < 0 or point[0] >= h or point[1] < 0 or point[1] >= w:
                continue
            if intersect_matrix[2, point[0], point[1]]:
                return 1

        if current_idx > 0:
            prev_point = traj[current_idx - 1]
            if prev_point[0] < 0 or prev_point[0] >= h or prev_point[1] < 0 or prev_point[1] >= w:
                return 0
            if intersect_matrix[2, prev_point[0], prev_point[1]]:
                return 1
        return 0
    else:
        if intersect_matrix[1, agent_position[0], agent_position[1]]:
            return 1
        else:
            return 0


def CarStepsToAtInter(world_matrix, intersect_matrix, agents, entity1, max_lookahead=3):
    if "PH" in entity1:
        return None
    _, agent_type, layer_id = entity1.split("_")
    if agent_type != "Car":
        return None
    layer_id = int(layer_id)
    agent_layer = world_matrix[layer_id]
    agent_position = (agent_layer == TYPE_MAP[agent_type]).nonzero()[0]

    if intersect_matrix[0, agent_position[0], agent_position[1]]:
        return 0

    agent = _get_agent_obj(agents, layer_id)
    traj = getattr(agent, "global_traj", None)
    if traj is not None and len(traj) > 0:
        matches = torch.all(traj == agent_position, dim=1).nonzero(as_tuple=False)
        if matches.numel() > 0:
            current_idx = int(matches[0].item())
            lookahead_limit = min(len(traj), current_idx + max_lookahead + 1)
            for idx in range(current_idx + 1, lookahead_limit):
                point = traj[idx]
                if (
                    0 <= point[0] < intersect_matrix.shape[1]
                    and 0 <= point[1] < intersect_matrix.shape[2]
                    and intersect_matrix[0, point[0], point[1]]
                ):
                    return idx - current_idx

    move_dir = _get_agent_direction(agents, str(layer_id))
    if move_dir is None:
        move_dir = _get_agent_direction(agents, layer_id)
    if move_dir is None:
        return None

    delta = torch.tensor(DIRECTION_VECTOR[move_dir], dtype=torch.long)
    pos = agent_position.clone().long()
    h, w = intersect_matrix.shape[1], intersect_matrix.shape[2]
    for step in range(1, max_lookahead + 1):
        pos = pos + delta
        if pos[0] < 0 or pos[0] >= h or pos[1] < 0 or pos[1] >= w:
            break
        if intersect_matrix[0, pos[0], pos[1]]:
            return step
    return None


def _intersection_context_id(world_matrix, intersect_matrix, agents, entity):
    if "PH" in entity:
        return 0
    _, agent_type, layer_id = entity.split("_")
    layer_id = int(layer_id)
    agent_layer = world_matrix[layer_id]
    agent_position = (agent_layer == TYPE_MAP[agent_type]).nonzero()[0]

    current_inter_id = int(intersect_matrix[2, agent_position[0], agent_position[1]].item())
    if current_inter_id > 0:
        return current_inter_id

    if agent_type == "Car":
        current_line_inter_id = int(intersect_matrix[0, agent_position[0], agent_position[1]].item())
    else:
        current_line_inter_id = int(intersect_matrix[1, agent_position[0], agent_position[1]].item())

    fallback_line_inter_id = current_line_inter_id if current_line_inter_id > 0 else 0

    if agent_type != "Car":
        agent_key = str(layer_id) if str(layer_id) in agents else "ego_{}".format(layer_id)
        if agent_key not in agents:
            return fallback_line_inter_id
    else:
        agent_key = str(layer_id) if str(layer_id) in agents else "ego_{}".format(layer_id)
        if agent_key not in agents:
            return fallback_line_inter_id

    agent = agents[agent_key]
    traj = getattr(agent, "global_traj", None)
    if traj is None or len(traj) == 0:
        return fallback_line_inter_id

    matches = torch.all(traj == agent_position, dim=1).nonzero(as_tuple=False)
    if matches.numel() == 0:
        return fallback_line_inter_id

    current_idx = int(matches[0].item())

    # Prefer the upcoming core intersection along the route. Returning the
    # current per-approach line id too early can split one physical junction
    # into separate contexts once IsAtInter is widened outward.
    h, w = intersect_matrix.shape[1], intersect_matrix.shape[2]
    lookahead_limit = min(len(traj), current_idx + 12)
    for idx in range(current_idx + 1, lookahead_limit):
        point = traj[idx]
        if point[0] < 0 or point[0] >= h or point[1] < 0 or point[1] >= w:
            continue
        inter_id = int(intersect_matrix[2, point[0], point[1]].item())
        if inter_id > 0:
            return inter_id

    lookbehind_start = max(0, current_idx - 3)
    for idx in range(current_idx - 1, lookbehind_start - 1, -1):
        point = traj[idx]
        if point[0] < 0 or point[0] >= h or point[1] < 0 or point[1] >= w:
            continue
        inter_id = int(intersect_matrix[2, point[0], point[1]].item())
        if inter_id > 0:
            return inter_id
    return fallback_line_inter_id


def _get_agent_direction(agents, layer_id):
    agent = None
    if layer_id in agents.keys():
        agent = agents[layer_id]
    else:
        ego_key = "ego_{}".format(layer_id)
        if ego_key in agents.keys():
            agent = agents[ego_key]
        else:
            for candidate in agents.values():
                if getattr(candidate, "layer_id", None) == layer_id:
                    agent = candidate
                    break
            assert agent is not None, "Agent {} not found in predicate context keys={}".format(layer_id, list(agents.keys()))
    if hasattr(agent, "moving_direction"):
        return agent.moving_direction
    return getattr(agent, "last_move_dir", None)


def _get_agent_obj(agents, layer_id):
    if layer_id in agents.keys():
        return agents[layer_id]
    ego_key = "ego_{}".format(layer_id)
    if ego_key in agents.keys():
        return agents[ego_key]
    for candidate in agents.values():
        if getattr(candidate, "layer_id", None) == layer_id:
            return candidate
    raise AssertionError("Agent {} not found in predicate context keys={}".format(layer_id, list(agents.keys())))


def _is_parked_car(agents, entity):
    if "PH" in entity:
        return False
    _, agent_type, layer_id = entity.split("_")
    if agent_type != "Car":
        return False
    agent = _get_agent_obj(agents, int(layer_id))
    if bool(getattr(agent, "reach_goal", False)):
        return True

    start = getattr(agent, "start", None)
    pos = getattr(agent, "pos", None)
    if start is None or pos is None:
        return False
    try:
        at_spawn = bool(torch.all(pos == start).item())
    except Exception:
        at_spawn = False
    return at_spawn


def IsParked(world_matrix, intersect_matrix, agents, entity1):
    return 1 if _is_parked_car(agents, entity1) else 0


def _agent_center(world_matrix, entity):
    _, agent_type, layer_id = entity.split("_")
    layer_id = int(layer_id)
    agent_layer = world_matrix[layer_id]
    return (agent_layer == TYPE_MAP[agent_type]).nonzero()[0]


def _car_future_centers(world_matrix, agents, entity, horizon_steps=2):
    _, agent_type, layer_id = entity.split("_")
    if agent_type != "Car":
        return [_agent_center(world_matrix, entity)]

    layer_id = int(layer_id)
    agent = _get_agent_obj(agents, layer_id)
    current_center = _agent_center(world_matrix, entity)
    traj = getattr(agent, "global_traj", None)
    if traj is None or len(traj) == 0:
        return [current_center]

    matches = torch.all(traj == current_center, dim=1).nonzero(as_tuple=False)
    if matches.numel() == 0:
        return [current_center]

    current_idx = int(matches[0].item())
    centers = [current_center]
    for step in range(1, horizon_steps + 1):
        next_idx = min(current_idx + step, len(traj) - 1)
        centers.append(traj[next_idx].clone())
    return centers


def _footprint_cells(center, radius, grid_shape):
    cx = int(center[0].item())
    cy = int(center[1].item())
    cells = set()
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            x = cx + dx
            y = cy + dy
            if 0 <= x < grid_shape[0] and 0 <= y < grid_shape[1]:
                cells.add((x, y))
    return cells


def _collision_footprint_radius(agent_type):
    """Use a slightly larger car footprint for collision predicates so the
    simulator's safety margin matches the rendered car size more closely."""
    if agent_type == "Car":
        return 2
    return 0


def _future_footprints_overlap(world_matrix, agents, entity1, entity2, horizon_steps=2):
    grid_shape = world_matrix.shape[1:]
    _, agent_type1, _ = entity1.split("_")
    _, agent_type2, _ = entity2.split("_")
    radius1 = _collision_footprint_radius(agent_type1)
    radius2 = _collision_footprint_radius(agent_type2)
    centers1 = _car_future_centers(world_matrix, agents, entity1, horizon_steps=horizon_steps)
    centers2 = _car_future_centers(world_matrix, agents, entity2, horizon_steps=horizon_steps)
    footprints1 = [_footprint_cells(center, radius1, grid_shape) for center in centers1]
    footprints2 = [_footprint_cells(center, radius2, grid_shape) for center in centers2]

    # Same-time imminent overlap.
    for cells1, cells2 in zip(footprints1, footprints2):
        if cells1.intersection(cells2):
            return 1

    # One-step crossing / swap conflict.
    max_t = min(len(footprints1), len(footprints2)) - 1
    for t in range(max_t):
        if footprints1[t + 1].intersection(footprints2[t]):
            return 1
        if footprints2[t + 1].intersection(footprints1[t]):
            return 1
    return 0


def _future_center_conflict(world_matrix, agents, entity1, entity2, horizon_steps=2):
    """Detect true short-horizon trajectory conflicts for car-car shielding.

    This is intentionally stricter than footprint overlap: it only flags cases
    where the two route centerlines would occupy the same cell at the same
    time, swap cells across one step, or merge into the same next route cell.
    That avoids freezing cars that are merely spatially close on different
    lanes of the same intersection arm.
    """
    centers1 = _car_future_centers(world_matrix, agents, entity1, horizon_steps=horizon_steps)
    centers2 = _car_future_centers(world_matrix, agents, entity2, horizon_steps=horizon_steps)

    min_len = min(len(centers1), len(centers2))
    for t in range(min_len):
        if torch.equal(centers1[t], centers2[t]):
            return 1

    max_t = min_len - 1
    for t in range(max_t):
        # Head-on swap / crossing over the same pair of cells.
        if torch.equal(centers1[t + 1], centers2[t]) and torch.equal(centers2[t + 1], centers1[t]):
            return 1
        # Merge conflict into the same next route cell.
        if torch.equal(centers1[t + 1], centers2[t + 1]):
            return 1
    return 0


def _current_center_conflict(world_matrix, agents, entity1, entity2):
    """Terminal car-car collision check aligned with route-conflict logic.

    For cars, a real collision should mean they occupy the same route cell
    after the transition, or they swapped cells across the last transition.
    This avoids false crashes from large icon-sized proximity on parallel or
    otherwise non-conflicting lanes.
    """
    center1 = _agent_center(world_matrix, entity1)
    center2 = _agent_center(world_matrix, entity2)
    if torch.equal(center1, center2):
        return 1

    _, agent_type1, layer_id1 = entity1.split("_")
    _, agent_type2, layer_id2 = entity2.split("_")
    if agent_type1 != "Car" or agent_type2 != "Car":
        return 0

    agent1 = _get_agent_obj(agents, int(layer_id1))
    agent2 = _get_agent_obj(agents, int(layer_id2))
    traj1 = getattr(agent1, "global_traj", None)
    traj2 = getattr(agent2, "global_traj", None)
    if traj1 is None or traj2 is None or len(traj1) == 0 or len(traj2) == 0:
        return 0

    matches1 = torch.all(traj1 == center1, dim=1).nonzero(as_tuple=False)
    matches2 = torch.all(traj2 == center2, dim=1).nonzero(as_tuple=False)
    if matches1.numel() == 0 or matches2.numel() == 0:
        return 0

    idx1 = int(matches1[0].item())
    idx2 = int(matches2[0].item())
    if idx1 > 0 and idx2 > 0:
        prev1 = traj1[idx1 - 1]
        prev2 = traj2[idx2 - 1]
        if torch.equal(prev1, center2) and torch.equal(prev2, center1):
            return 1
    return 0


def _same_lane_front_gap_conflict(world_matrix, agents, entity1, entity2, max_gap_cells):
    """Detect rear-end risk for cars on the same lane and same heading."""
    _, agent_type1, layer_id1 = entity1.split("_")
    _, agent_type2, layer_id2 = entity2.split("_")
    if agent_type1 != "Car" or agent_type2 != "Car":
        return 0

    center1 = _agent_center(world_matrix, entity1).to(torch.int64)
    center2 = _agent_center(world_matrix, entity2).to(torch.int64)
    dir1 = _get_agent_direction(agents, layer_id1)
    dir2 = _get_agent_direction(agents, layer_id2)
    if dir1 is None or dir2 is None or dir1 != dir2:
        return 0

    facing = torch.tensor(DIRECTION_VECTOR[dir1], dtype=torch.int64)

    def _is_ahead(relative_vec):
        forward_gap = int(torch.dot(relative_vec, facing).item())
        lateral_gap = int(abs(relative_vec[0].item() * facing[1].item() - relative_vec[1].item() * facing[0].item()))
        return lateral_gap == 0 and 0 < forward_gap <= max_gap_cells

    if _is_ahead(center2 - center1):
        return 1
    if _is_ahead(center1 - center2):
        return 1
    return 0

def _current_footprints_overlap(world_matrix, entity1, entity2):
    grid_shape = world_matrix.shape[1:]
    _, agent_type1, _ = entity1.split("_")
    _, agent_type2, _ = entity2.split("_")
    center1 = _agent_center(world_matrix, entity1)
    center2 = _agent_center(world_matrix, entity2)
    radius1 = _collision_footprint_radius(agent_type1)
    radius2 = _collision_footprint_radius(agent_type2)
    footprint1 = _footprint_cells(center1, radius1, grid_shape)
    footprint2 = _footprint_cells(center2, radius2, grid_shape)
    if footprint1.intersection(footprint2):
        return 1
    return 0

def IsInInter(world_matrix, intersect_matrix, agents, entity1):
    if "PH" in entity1:
        return 0
    _, agent_type, layer_id = entity1.split("_")
    layer_id = int(layer_id)
    agent_layer = world_matrix[layer_id]
    agent_position = (agent_layer == TYPE_MAP[agent_type]).nonzero()[0]
    if intersect_matrix[2, agent_position[0], agent_position[1]]:
        return 1
    else:
        return 0


def SameInter(world_matrix, intersect_matrix, agents, entity1, entity2):
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    inter_id_1 = _intersection_context_id(world_matrix, intersect_matrix, agents, entity1)
    inter_id_2 = _intersection_context_id(world_matrix, intersect_matrix, agents, entity2)
    if inter_id_1 > 0 and inter_id_1 == inter_id_2:
        return 1
    return 0

def IsClose(world_matrix, intersect_matrix, agents, entity1, entity2):
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    _, agent_type1, layer_id1 = entity1.split("_")
    _, agent_type2, layer_id2 = entity2.split("_")
    layer_id1 = int(layer_id1)
    layer_id2 = int(layer_id2)
    agent_layer1 = world_matrix[layer_id1]
    agent_layer2 = world_matrix[layer_id2]
    agent_position1 = (agent_layer1 == TYPE_MAP[agent_type1]).nonzero()[0]
    agent_position2 = (agent_layer2 == TYPE_MAP[agent_type2]).nonzero()[0]
    eudis = torch.sqrt(torch.sum((agent_position1 - agent_position2)**2))
    if eudis > CLOSE_RANGE_MIN and eudis <= CLOSE_RANGE_MAX:
        return 1
    else:
        return 0

def HigherPri(world_matrix, intersect_matrix, agents, entity1, entity2):
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    
    _, _, agent_layer1 = entity1.split("_")
    _, _, agent_layer2 = entity2.split("_")

    if agent_layer1 in agents.keys():
        agent_prio1 = agents[agent_layer1].priority
    else:
        assert "ego_{}".format(agent_layer1) in agents.keys()
        agent_prio1 = agents["ego_{}".format(agent_layer1)].priority

    if agent_layer2 in agents.keys():
        agent_prio2 = agents[agent_layer2].priority
    else:
        assert "ego_{}".format(agent_layer2) in agents.keys()
        agent_prio2 = agents["ego_{}".format(agent_layer2)].priority

    if agent_prio1 < agent_prio2:
        return 1
    else:
        return 0

def CollidingClose(world_matrix, intersect_matrix, agents, entity1, entity2):
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    if _is_parked_car(agents, entity1) or _is_parked_car(agents, entity2):
        return 0
    _, agent_type1, layer_id1 = entity1.split("_")
    _, agent_type2, layer_id2 = entity2.split("_")

    if agent_type1 == "Car" and agent_type2 == "Car":
        if _same_lane_front_gap_conflict(world_matrix, agents, entity1, entity2, max_gap_cells=12):
            return 1
        same_inter = SameInter(world_matrix, intersect_matrix, agents, entity1, entity2)
        close_geom = IsClose(world_matrix, intersect_matrix, agents, entity1, entity2)
        if not same_inter and not close_geom:
            return 0
        return _future_center_conflict(world_matrix, agents, entity1, entity2, horizon_steps=2)

    agent_position1 = _agent_center(world_matrix, entity1)
    agent_position2 = _agent_center(world_matrix, entity2)
    agent1_dire = _get_agent_direction(agents, layer_id1)
    if agent1_dire == None:
        return 0

    dist = torch.sqrt(torch.sum((agent_position1 - agent_position2)**2))
    if dist > OCC_CHECK_RANGE[agent_type1]:
        return 0
    elif dist == 0:
        return np.random.choice([0, 1], p=[0.5, 0.5])
    else:
        agent1_dire_vec = torch.tensor(DIRECTION_VECTOR[agent1_dire], dtype=torch.float32)
        relative_vec = (agent_position2 - agent_position1).to(torch.float32)
        cos_angle = torch.dot(agent1_dire_vec, relative_vec) / dist
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        angle = torch.acos(cos_angle)
        if angle < OCC_CHECK_ANGEL:
            if agent_type1 == "Car":
                return 1
            else:
                sample = np.random.rand()
                if sample < PED_AGGR:
                    return 1
                else:
                    return 0
    return 0

def Collision(world_matrix, intersect_matrix, agents, entity1, entity2):
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    if _is_parked_car(agents, entity1) or _is_parked_car(agents, entity2):
        return 0
    _, agent_type1, _ = entity1.split("_")
    _, agent_type2, _ = entity2.split("_")
    if agent_type1 == "Car" and agent_type2 == "Car":
        if _same_lane_front_gap_conflict(world_matrix, agents, entity1, entity2, max_gap_cells=8):
            return 1
        return _current_center_conflict(world_matrix, agents, entity1, entity2)
    return _current_footprints_overlap(world_matrix, entity1, entity2)

def LeftOf(world_matrix, intersect_matrix, agents, entity1, entity2):
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    # 1. Get the position of the two agents
    _, agent_type1, layer_id1 = entity1.split("_")
    _, agent_type2, layer_id2 = entity2.split("_")
    agent_layer1 = world_matrix[int(layer_id1)]
    agent_layer2 = world_matrix[int(layer_id2)]
    agent_position1 = (agent_layer1 == TYPE_MAP[agent_type1]).nonzero()[0]
    agent_position2 = (agent_layer2 == TYPE_MAP[agent_type2]).nonzero()[0]
    # 2. note: entity1 is on the left of entity2, so we get the direction of entity2
    if layer_id2 in agents.keys():
        agent2_dire = agents[layer_id2].moving_direction
    else:
        assert "ego_{}".format(layer_id2) in agents.keys()
        agent2_dire = agents["ego_{}".format(layer_id2)].moving_direction
    if agent2_dire == None:
        return 0
    else:
        agent2_dire_vec = torch.tensor(DIRECTION_VECTOR[agent2_dire])
        relative_pos = agent_position1 - agent_position2
        dx, dy = agent2_dire_vec # Direction Agent 2 is facing
        rx, ry = relative_pos # Vector from Agent 2 to Agent 1
        z_component = dx * ry - dy * rx
        if z_component > 0:
            return 1
        else:
            return 0

def RightOf(world_matrix, intersect_matrix, agents, entity1, entity2):
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    # 1. Get the position of the two agents
    _, agent_type1, layer_id1 = entity1.split("_")
    _, agent_type2, layer_id2 = entity2.split("_")
    agent_layer1 = world_matrix[int(layer_id1)]
    agent_layer2 = world_matrix[int(layer_id2)]
    agent_position1 = (agent_layer1 == TYPE_MAP[agent_type1]).nonzero()[0]
    agent_position2 = (agent_layer2 == TYPE_MAP[agent_type2]).nonzero()[0]
    # 2. note: entity1 is on the right of entity2, so we get the direction of entity2
    if layer_id2 in agents.keys():
        agent2_dire = agents[layer_id2].moving_direction
    else:
        assert "ego_{}".format(layer_id2) in agents.keys()
        agent2_dire = agents["ego_{}".format(layer_id2)].moving_direction
    if agent2_dire == None:
        return 0
    else:
        agent2_dire_vec = torch.tensor(DIRECTION_VECTOR[agent2_dire])
        relative_pos = agent_position1 - agent_position2
        dx, dy = agent2_dire_vec # Direction Agent 2 is facing
        rx, ry = relative_pos # Vector from Agent 2 to Agent 1
        z_component = dx * ry - dy * rx
        if z_component < 0:
            return 1
        else:
            return 0

def NextTo(world_matrix, intersect_matrix, agents, entity1, entity2):
    # TODO: Next to checker
    # Next to checker, closer than Close checker
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    _, agent_type1, layer_id1 = entity1.split("_")
    _, agent_type2, layer_id2 = entity2.split("_")
    layer_id1 = int(layer_id1)
    layer_id2 = int(layer_id2)
    agent_layer1 = world_matrix[layer_id1]
    agent_layer2 = world_matrix[layer_id2]
    agent_position1 = (agent_layer1 == TYPE_MAP[agent_type1]).nonzero()[0]
    agent_position2 = (agent_layer2 == TYPE_MAP[agent_type2]).nonzero()[0]
    eudis = torch.sqrt(torch.sum((agent_position1 - agent_position2)**2))
    if eudis < CLOSE_RANGE_MIN:
        return 1
    else:
        return 0
