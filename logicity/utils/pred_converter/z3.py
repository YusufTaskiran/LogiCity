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
    # at intersection needs to care if the car is "entering" or "leaving", so use intersect_matrix[0]
    if agent_type == "Car":
        if intersect_matrix[0, agent_position[0], agent_position[1]]:
            return 1
        else:
            return 0
    else:
        if intersect_matrix[1, agent_position[0], agent_position[1]]:
            return 1
        else:
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

def IsNear(world_matrix, intersect_matrix, agents, entity1, entity2):
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    _, agent_type1, layer_id1 = entity1.split("_")
    _, agent_type2, layer_id2 = entity2.split("_")
    agent_layer1 = world_matrix[int(layer_id1)]
    agent_layer2 = world_matrix[int(layer_id2)]
    agent_position1 = (agent_layer1 == TYPE_MAP[agent_type1]).nonzero()[0]
    agent_position2 = (agent_layer2 == TYPE_MAP[agent_type2]).nonzero()[0]
    dist = torch.sqrt(torch.sum((agent_position1 - agent_position2) ** 2))
    if dist <= OCC_CHECK_RANGE[agent_type1]:
        return 1
    return 0

def IsAhead(world_matrix, intersect_matrix, agents, entity1, entity2):
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    _, agent_type1, layer_id1 = entity1.split("_")
    _, agent_type2, layer_id2 = entity2.split("_")
    if agent_type1 != "Car":
        return 0
    if agent_type2 != "Car":
        return 0
    agent_layer1 = world_matrix[int(layer_id1)]
    agent_layer2 = world_matrix[int(layer_id2)]
    agent_position1 = (agent_layer1 == TYPE_MAP[agent_type1]).nonzero()[0]
    agent_position2 = (agent_layer2 == TYPE_MAP[agent_type2]).nonzero()[0]
    agent1_record = _get_agent_record_any(agents, layer_id1)
    agent1_dire = _infer_moving_direction(agent_layer1, agent_type1, agent_position1, _get_agent_direction(agent1_record))
    if agent1_dire is None:
        return 0
    dist = _forward_lane_distance(agent_position1, agent_position2, agent1_dire)
    if dist is None:
        return 0
    min_ahead_range = 3.0
    max_ahead_range = min(4.0, _forward_visibility_limit(agent_position1, agent1_dire, agent_layer1.shape))
    return int(min_ahead_range <= dist <= max_ahead_range)

def IsCloseAhead(world_matrix, intersect_matrix, agents, entity1, entity2):
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    _, agent_type1, layer_id1 = entity1.split("_")
    _, agent_type2, layer_id2 = entity2.split("_")
    if agent_type1 != "Car":
        return 0
    if agent_type2 != "Car":
        return 0
    agent_layer1 = world_matrix[int(layer_id1)]
    agent_layer2 = world_matrix[int(layer_id2)]
    agent_position1 = (agent_layer1 == TYPE_MAP[agent_type1]).nonzero()[0]
    agent_position2 = (agent_layer2 == TYPE_MAP[agent_type2]).nonzero()[0]
    agent1_record = _get_agent_record_any(agents, layer_id1)
    agent1_dire = _infer_moving_direction(agent_layer1, agent_type1, agent_position1, _get_agent_direction(agent1_record))
    if agent1_dire is None:
        return 0
    dist = _forward_corridor_distance(agent_position1, agent_position2, agent1_dire, lateral_tolerance=1)
    if dist is None:
        return 0
    close_min = 3.0
    close_max = min(3.0, _forward_visibility_limit(agent_position1, agent1_dire, agent_layer1.shape))
    return int(close_min <= dist <= close_max)

def HigherPri(world_matrix, intersect_matrix, agents, entity1, entity2):
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    
    _, _, agent_layer1 = entity1.split("_")
    _, _, agent_layer2 = entity2.split("_")

    agent_prio1 = _get_agent_record_any(agents, agent_layer1).priority
    agent_prio2 = _get_agent_record_any(agents, agent_layer2).priority

    if agent_prio1 < agent_prio2:
        return 1
    else:
        return 0

def _get_agent_record(agents, layer_id):
    if layer_id in agents:
        return agents[layer_id]
    ego_key = f"ego_{layer_id}"
    if ego_key in agents:
        return agents[ego_key]
    raise KeyError(f"Agent layer {layer_id} not found in partial agent map.")

def _get_agent_record_any(agents, layer_id):
    if isinstance(agents, dict):
        return _get_agent_record(agents, layer_id)
    for agent in agents:
        if str(agent.layer_id) == str(layer_id):
            return agent
    raise KeyError(f"Agent layer {layer_id} not found.")

def _get_agent_direction(agent_record):
    direction = getattr(agent_record, "moving_direction", None)
    if direction is not None:
        return direction
    return getattr(agent_record, "last_move_dir", None)

def _get_entity_state(world_matrix, agents, entity):
    _, agent_type, layer_id = entity.split("_")
    agent_layer = world_matrix[int(layer_id)]
    agent_position = (agent_layer == TYPE_MAP[agent_type]).nonzero()[0]
    agent_record = _get_agent_record_any(agents, layer_id)
    return agent_type, layer_id, agent_layer, agent_position, agent_record

def _infer_moving_direction(agent_layer, agent_type, agent_position, current_direction):
    if current_direction is not None:
        return current_direction
    path_value = TYPE_MAP[agent_type] + AGENT_GLOBAL_PATH_PLUS
    goal_value = TYPE_MAP[agent_type] + AGENT_GOAL_PLUS
    rows, cols = agent_layer.shape
    best_direction = None
    best_step = None
    for direction, delta in DIRECTION_VECTOR.items():
        move = torch.tensor(delta)
        for step in range(1, 4):
            next_pos = agent_position + move * step
            x, y = int(next_pos[0]), int(next_pos[1])
            if x < 0 or x >= rows or y < 0 or y >= cols:
                break
            value = float(agent_layer[x, y].item())
            if abs(value - path_value) < 1e-4 or abs(value - goal_value) < 1e-4:
                if best_step is None or step < best_step:
                    best_direction = direction
                    best_step = step
                break
    return best_direction

def _forward_visibility_limit(agent_position, direction, grid_shape):
    rows, cols = grid_shape
    x, y = int(agent_position[0]), int(agent_position[1])
    if direction == "Left":
        return float(y)
    if direction == "Right":
        return float(cols - 1 - y)
    if direction == "Up":
        return float(x)
    if direction == "Down":
        return float(rows - 1 - x)
    return 0.0

def _forward_lane_distance(agent_position1, agent_position2, direction):
    x1, y1 = int(agent_position1[0]), int(agent_position1[1])
    x2, y2 = int(agent_position2[0]), int(agent_position2[1])
    if direction == "Left":
        if x1 != x2 or y2 >= y1:
            return None
        return float(y1 - y2)
    if direction == "Right":
        if x1 != x2 or y2 <= y1:
            return None
        return float(y2 - y1)
    if direction == "Up":
        if y1 != y2 or x2 >= x1:
            return None
        return float(x1 - x2)
    if direction == "Down":
        if y1 != y2 or x2 <= x1:
            return None
        return float(x2 - x1)
    return None

def _forward_corridor_distance(agent_position1, agent_position2, direction, lateral_tolerance=1):
    x1, y1 = int(agent_position1[0]), int(agent_position1[1])
    x2, y2 = int(agent_position2[0]), int(agent_position2[1])
    if direction == "Left":
        if y2 >= y1 or abs(x2 - x1) > lateral_tolerance:
            return None
        return float(y1 - y2)
    if direction == "Right":
        if y2 <= y1 or abs(x2 - x1) > lateral_tolerance:
            return None
        return float(y2 - y1)
    if direction == "Up":
        if x2 >= x1 or abs(y2 - y1) > lateral_tolerance:
            return None
        return float(x1 - x2)
    if direction == "Down":
        if x2 <= x1 or abs(y2 - y1) > lateral_tolerance:
            return None
        return float(x2 - x1)
    return None

def _in_forward_deadzone(agent_position1, agent_position2, direction):
    x1, y1 = int(agent_position1[0]), int(agent_position1[1])
    x2, y2 = int(agent_position2[0]), int(agent_position2[1])
    if direction == "Left":
        return (
            (x2 == x1 and y2 in {y1 - 1, y1 - 2})
            or (y2 == y1 - 1 and x2 in {x1 - 1, x1 + 1})
        )
    if direction == "Right":
        return (
            (x2 == x1 and y2 in {y1 + 1, y1 + 2})
            or (y2 == y1 + 1 and x2 in {x1 - 1, x1 + 1})
        )
    if direction == "Up":
        return (
            (y2 == y1 and x2 in {x1 - 1, x1 - 2})
            or (x2 == x1 - 1 and y2 in {y1 - 1, y1 + 1})
        )
    if direction == "Down":
        return (
            (y2 == y1 and x2 in {x1 + 1, x1 + 2})
            or (x2 == x1 + 1 and y2 in {y1 - 1, y1 + 1})
        )
    return False

def _swept_cells(agent_position, direction, step_count, grid_shape):
    if direction is None or step_count <= 0:
        return []
    rows, cols = grid_shape
    delta = torch.tensor(DIRECTION_VECTOR[direction])
    cells = []
    current = agent_position.clone()
    for _ in range(step_count):
        current = current + delta
        x, y = int(current[0]), int(current[1])
        if x < 0 or x >= rows or y < 0 or y >= cols:
            break
        cells.append((x, y))
    return cells

def _projected_cells(world_matrix, agents, entity, default_step_count):
    agent_type, _, agent_layer, agent_position, agent_record = _get_entity_state(world_matrix, agents, entity)
    direction = _infer_moving_direction(agent_layer, agent_type, agent_position, _get_agent_direction(agent_record))
    swept = _swept_cells(agent_position, direction, default_step_count, agent_layer.shape)
    occupied = {(int(agent_position[0]), int(agent_position[1]))}
    occupied.update(swept)
    return occupied, direction

def _priority_value(agent_record):
    return int(agent_record.priority)

def _has_strict_precedence(other_record, ego_record):
    other_priority = _priority_value(other_record)
    ego_priority = _priority_value(ego_record)
    if other_priority != ego_priority:
        return other_priority < ego_priority
    return int(other_record.layer_id) < int(ego_record.layer_id)

def _intersection_mask(intersect_matrix):
    return torch.any(intersect_matrix > 0, dim=0)

def _intersection_interior_mask(intersect_matrix):
    return intersect_matrix[2] > 0

def _intersection_entry_mask(intersect_matrix, agent_type):
    if agent_type == "Car":
        return intersect_matrix[0] > 0
    return intersect_matrix[1] > 0

def _entity_position(world_matrix, entity):
    _, agent_type, layer_id = entity.split("_")
    agent_layer = world_matrix[int(layer_id)]
    return agent_type, (agent_layer == TYPE_MAP[agent_type]).nonzero()[0]

def _entry_intersection_block_id(world_matrix, intersect_matrix, entity):
    if "PH" in entity:
        return 0
    agent_type, agent_position = _entity_position(world_matrix, entity)
    layer_idx = 0 if agent_type == "Car" else 1
    return int(intersect_matrix[layer_idx, int(agent_position[0]), int(agent_position[1])].item())

def _interior_intersection_block_id(world_matrix, intersect_matrix, entity):
    if "PH" in entity:
        return 0
    _, agent_position = _entity_position(world_matrix, entity)
    return int(intersect_matrix[2, int(agent_position[0]), int(agent_position[1])].item())

def global_in_intersection_higher_pri_conflict(world_matrix, intersect_matrix, agents, entity):
    if "PH" in entity:
        return 0
    _, agent_type, layer_id = entity.split("_")
    if agent_type != "Car":
        return 0
    agent_layer = world_matrix[int(layer_id)]
    agent_position = (agent_layer == TYPE_MAP[agent_type]).nonzero()[0]
    ego_block_id = _interior_intersection_block_id(world_matrix, intersect_matrix, entity)
    if ego_block_id <= 0:
        return 0
    ego_record = _get_agent_record_any(agents, layer_id)
    if isinstance(agents, dict):
        agent_iter = agents.values()
    else:
        agent_iter = agents
    seen_layer_ids = set()
    for other_record in agent_iter:
        other_layer_id = str(other_record.layer_id)
        if other_layer_id in seen_layer_ids:
            continue
        seen_layer_ids.add(other_layer_id)
        if other_layer_id == str(layer_id):
            continue
        if getattr(other_record, "type", None) == "PH":
            continue
        if getattr(other_record, "type", None) != "Car":
            continue
        other_agent_layer = world_matrix[int(other_layer_id)]
        other_position = (other_agent_layer == TYPE_MAP[other_record.type]).nonzero()[0]
        other_block_id = _interior_intersection_block_id(
            world_matrix,
            intersect_matrix,
            f"Entity_{other_record.type}_{other_layer_id}",
        )
        if other_block_id != ego_block_id:
            continue
        if _has_strict_precedence(other_record, ego_record):
            return 1
    return 0

def global_stop_required_conflict(world_matrix, intersect_matrix, agents, entity):
    if "PH" in entity:
        return 0

    if isinstance(agents, dict):
        agent_iter = agents.values()
    else:
        agent_iter = agents

    seen_layer_ids = set()
    other_entities = []
    for other_record in agent_iter:
        other_layer_id = str(other_record.layer_id)
        if other_layer_id in seen_layer_ids:
            continue
        seen_layer_ids.add(other_layer_id)
        other_type = getattr(other_record, "type", None)
        if other_type in (None, "PH"):
            continue
        other_entities.append(f"Entity_{other_type}_{other_layer_id}")

    for other_entity in other_entities:
        if other_entity == entity:
            continue

        ego_entry_block = _entry_intersection_block_id(world_matrix, intersect_matrix, entity)
        other_entry_block = _entry_intersection_block_id(world_matrix, intersect_matrix, other_entity)
        ego_interior_block = _interior_intersection_block_id(world_matrix, intersect_matrix, entity)
        other_interior_block = _interior_intersection_block_id(world_matrix, intersect_matrix, other_entity)

        if ego_entry_block > 0 and ego_entry_block == other_interior_block:
            return 1

        if (
            ego_entry_block > 0
            and ego_entry_block == other_entry_block
            and HigherPri(world_matrix, intersect_matrix, agents, other_entity, entity)
        ):
            return 1

        if (
            IsCar(world_matrix, intersect_matrix, agents, entity)
            and ego_interior_block > 0
            and IsCar(world_matrix, intersect_matrix, agents, other_entity)
            and ego_interior_block == other_interior_block
            and HigherPri(world_matrix, intersect_matrix, agents, other_entity, entity)
        ):
            return 1

        if IsCloseAhead(world_matrix, intersect_matrix, agents, entity, other_entity):
            return 1

        if CollidingClose(world_matrix, intersect_matrix, agents, entity, other_entity):
            return 1

    return 0


def global_stop_required_conflict_witness(world_matrix, intersect_matrix, agents, entity):
    if "PH" in entity:
        return None

    if isinstance(agents, dict):
        agent_iter = agents.values()
    else:
        agent_iter = agents

    seen_layer_ids = set()
    other_entities = []
    for other_record in agent_iter:
        other_layer_id = str(other_record.layer_id)
        if other_layer_id in seen_layer_ids:
            continue
        seen_layer_ids.add(other_layer_id)
        other_type = getattr(other_record, "type", None)
        if other_type in (None, "PH"):
            continue
        other_entities.append(f"Entity_{other_type}_{other_layer_id}")

    for other_entity in other_entities:
        if other_entity == entity:
            continue

        ego_entry_block = _entry_intersection_block_id(world_matrix, intersect_matrix, entity)
        other_entry_block = _entry_intersection_block_id(world_matrix, intersect_matrix, other_entity)
        ego_interior_block = _interior_intersection_block_id(world_matrix, intersect_matrix, entity)
        other_interior_block = _interior_intersection_block_id(world_matrix, intersect_matrix, other_entity)

        if ego_entry_block > 0 and ego_entry_block == other_interior_block:
            return {"clause": "occupied_intersection_entry", "other_entity": other_entity}

        if (
            ego_entry_block > 0
            and ego_entry_block == other_entry_block
            and HigherPri(world_matrix, intersect_matrix, agents, other_entity, entity)
        ):
            return {"clause": "higher_priority_entry", "other_entity": other_entity}

        if (
            IsCar(world_matrix, intersect_matrix, agents, entity)
            and ego_interior_block > 0
            and IsCar(world_matrix, intersect_matrix, agents, other_entity)
            and ego_interior_block == other_interior_block
            and HigherPri(world_matrix, intersect_matrix, agents, other_entity, entity)
        ):
            return {"clause": "higher_priority_in_intersection", "other_entity": other_entity}

        if IsCloseAhead(world_matrix, intersect_matrix, agents, entity, other_entity):
            return {"clause": "close_ahead", "other_entity": other_entity}

        if CollidingClose(world_matrix, intersect_matrix, agents, entity, other_entity):
            return {"clause": "colliding_close", "other_entity": other_entity}

    return None

def global_soft_stop_required_conflict(world_matrix, intersect_matrix, agents, entity):
    if "PH" in entity:
        return 0

    if isinstance(agents, dict):
        agent_iter = agents.values()
    else:
        agent_iter = agents

    seen_layer_ids = set()
    other_entities = []
    for other_record in agent_iter:
        other_layer_id = str(other_record.layer_id)
        if other_layer_id in seen_layer_ids:
            continue
        seen_layer_ids.add(other_layer_id)
        other_type = getattr(other_record, "type", None)
        if other_type in (None, "PH"):
            continue
        other_entities.append(f"Entity_{other_type}_{other_layer_id}")

    for other_entity in other_entities:
        if other_entity == entity:
            continue

        ego_entry_block = _entry_intersection_block_id(world_matrix, intersect_matrix, entity)
        other_entry_block = _entry_intersection_block_id(world_matrix, intersect_matrix, other_entity)
        ego_interior_block = _interior_intersection_block_id(world_matrix, intersect_matrix, entity)
        other_interior_block = _interior_intersection_block_id(world_matrix, intersect_matrix, other_entity)

        if ego_entry_block > 0 and ego_entry_block == other_interior_block:
            return 1

        if (
            ego_entry_block > 0
            and ego_entry_block == other_entry_block
            and HigherPri(world_matrix, intersect_matrix, agents, other_entity, entity)
        ):
            return 1

        if (
            IsCar(world_matrix, intersect_matrix, agents, entity)
            and ego_interior_block > 0
            and IsCar(world_matrix, intersect_matrix, agents, other_entity)
            and ego_interior_block == other_interior_block
            and HigherPri(world_matrix, intersect_matrix, agents, other_entity, entity)
        ):
            return 1

    return 0

def global_soft_stop_violation_count(world_matrix, intersect_matrix, agents, entity):
    breakdown = global_soft_stop_violation_breakdown(world_matrix, intersect_matrix, agents, entity)
    return int(
        breakdown["occupied_entry"]
        + breakdown["higher_pri_entry"]
        + breakdown["higher_pri_in_inter"]
    )

def global_soft_stop_violation_breakdown(world_matrix, intersect_matrix, agents, entity):
    if "PH" in entity:
        return {
            "occupied_entry": 0,
            "higher_pri_entry": 0,
            "higher_pri_in_inter": 0,
        }

    if isinstance(agents, dict):
        agent_iter = agents.values()
    else:
        agent_iter = agents

    seen_layer_ids = set()
    other_entities = []
    for other_record in agent_iter:
        other_layer_id = str(other_record.layer_id)
        if other_layer_id in seen_layer_ids:
            continue
        seen_layer_ids.add(other_layer_id)
        other_type = getattr(other_record, "type", None)
        if other_type in (None, "PH"):
            continue
        other_entities.append(f"Entity_{other_type}_{other_layer_id}")

    occupied_entry = 0
    higher_pri_entry = 0
    higher_pri_in_inter = 0
    for other_entity in other_entities:
        if other_entity == entity:
            continue

        ego_entry_block = _entry_intersection_block_id(world_matrix, intersect_matrix, entity)
        other_entry_block = _entry_intersection_block_id(world_matrix, intersect_matrix, other_entity)
        ego_interior_block = _interior_intersection_block_id(world_matrix, intersect_matrix, entity)
        other_interior_block = _interior_intersection_block_id(world_matrix, intersect_matrix, other_entity)

        if ego_entry_block > 0 and ego_entry_block == other_interior_block:
            occupied_entry = 1

        if (
            ego_entry_block > 0
            and ego_entry_block == other_entry_block
            and HigherPri(world_matrix, intersect_matrix, agents, other_entity, entity)
        ):
            higher_pri_entry = 1

        if (
            IsCar(world_matrix, intersect_matrix, agents, entity)
            and ego_interior_block > 0
            and IsCar(world_matrix, intersect_matrix, agents, other_entity)
            and ego_interior_block == other_interior_block
            and HigherPri(world_matrix, intersect_matrix, agents, other_entity, entity)
        ):
            higher_pri_in_inter = 1

    return {
        "occupied_entry": int(occupied_entry),
        "higher_pri_entry": int(higher_pri_entry),
        "higher_pri_in_inter": int(higher_pri_in_inter),
    }

def global_deadzone_stop_required_conflict(world_matrix, intersect_matrix, agents, entity):
    if "PH" in entity:
        return 0

    if isinstance(agents, dict):
        agent_iter = agents.values()
    else:
        agent_iter = agents

    seen_layer_ids = set()
    for other_record in agent_iter:
        other_layer_id = str(other_record.layer_id)
        if other_layer_id in seen_layer_ids:
            continue
        seen_layer_ids.add(other_layer_id)
        other_type = getattr(other_record, "type", None)
        if other_type in (None, "PH"):
            continue
        other_entity = f"Entity_{other_type}_{other_layer_id}"
        if other_entity == entity:
            continue
        if CollidingClose(world_matrix, intersect_matrix, agents, entity, other_entity):
            return 1

    return 0

def _real_entity_names_from_agents(agents):
    if isinstance(agents, dict):
        agent_iter = agents.values()
    else:
        agent_iter = agents
    seen_layer_ids = set()
    entities = []
    for record in agent_iter:
        layer_id = str(record.layer_id)
        if layer_id in seen_layer_ids:
            continue
        seen_layer_ids.add(layer_id)
        agent_type = getattr(record, "type", None)
        if agent_type in (None, "PH"):
            continue
        entities.append(f"Entity_{agent_type}_{layer_id}")
    return entities

def _interior_intersection_block_id_from_world(world_matrix, intersect_matrix, entity):
    if "PH" in entity:
        return 0
    _, agent_type, layer_id = entity.split("_")
    layer_id = int(layer_id)
    agent_layer = world_matrix[layer_id]
    agent_positions = (agent_layer == TYPE_MAP[agent_type]).nonzero()
    if len(agent_positions) == 0:
        return 0
    agent_position = agent_positions[0]
    return int(intersect_matrix[2, int(agent_position[0]), int(agent_position[1])].item())

def global_deadzone_entry_failure(prev_world_matrix, world_matrix, intersect_matrix, agents):
    entities = _real_entity_names_from_agents(agents)
    for entity in entities:
        for other_entity in entities:
            if entity == other_entity:
                continue
            current_conflict = CollidingClose(world_matrix, intersect_matrix, agents, entity, other_entity)
            previous_conflict = CollidingClose(prev_world_matrix, intersect_matrix, agents, entity, other_entity)
            if current_conflict and not previous_conflict:
                return 1
    return 0

def global_simultaneous_intersection_entry_failure(prev_world_matrix, world_matrix, intersect_matrix, agents):
    entities = _real_entity_names_from_agents(agents)
    entered_per_block = {}
    for entity in entities:
        prev_block = _interior_intersection_block_id_from_world(prev_world_matrix, intersect_matrix, entity)
        current_block = _interior_intersection_block_id_from_world(world_matrix, intersect_matrix, entity)
        if prev_block == 0 and current_block > 0:
            entered_per_block[current_block] = entered_per_block.get(current_block, 0) + 1
    return int(any(count >= 2 for count in entered_per_block.values()))

def global_hard_failure_breakdown(prev_world_matrix, world_matrix, intersect_matrix, agents):
    deadzone = int(global_deadzone_entry_failure(prev_world_matrix, world_matrix, intersect_matrix, agents))
    simultaneous_entry = int(
        global_simultaneous_intersection_entry_failure(
            prev_world_matrix,
            world_matrix,
            intersect_matrix,
            agents,
        )
    )
    return {
        "deadzone": deadzone,
        "simultaneous_entry": simultaneous_entry,
    }

def global_close_ahead_stop_required_conflict(world_matrix, intersect_matrix, agents, entity):
    if "PH" in entity:
        return 0

    if isinstance(agents, dict):
        agent_iter = agents.values()
    else:
        agent_iter = agents

    seen_layer_ids = set()
    for other_record in agent_iter:
        other_layer_id = str(other_record.layer_id)
        if other_layer_id in seen_layer_ids:
            continue
        seen_layer_ids.add(other_layer_id)
        other_type = getattr(other_record, "type", None)
        if other_type in (None, "PH"):
            continue
        other_entity = f"Entity_{other_type}_{other_layer_id}"
        if other_entity == entity:
            continue
        if IsCloseAhead(world_matrix, intersect_matrix, agents, entity, other_entity):
            return 1

    return 0

def _pairwise_step_conflict(world_matrix, intersect_matrix, agents, entity, other_entity, step_count):
    if "PH" in entity or "PH" in other_entity or entity == other_entity:
        return 0
    agent_type, layer_id, agent_layer, agent_position, agent_record = _get_entity_state(world_matrix, agents, entity)
    other_type, _, _, other_position, other_record = _get_entity_state(world_matrix, agents, other_entity)
    direction = _infer_moving_direction(agent_layer, agent_type, agent_position, agent_record.moving_direction)
    if direction is None:
        return 0

    ego_swept = _swept_cells(agent_position, direction, step_count, agent_layer.shape)
    if len(ego_swept) == 0:
        return 0

    intersection_mask = _intersection_mask(intersect_matrix)
    interior_intersection_mask = _intersection_interior_mask(intersect_matrix)
    entry_intersection_mask = _intersection_entry_mask(intersect_matrix, agent_type)
    ego_final_cell = ego_swept[-1]
    ego_in_intersection = bool(interior_intersection_mask[int(agent_position[0]), int(agent_position[1])])
    ego_at_intersection_after = bool(entry_intersection_mask[ego_final_cell[0], ego_final_cell[1]])
    ego_in_intersection_after = (
        bool(interior_intersection_mask[ego_final_cell[0], ego_final_cell[1]])
        or any(interior_intersection_mask[x, y] for x, y in ego_swept)
    )

    other_step_count = 3 if other_type == "Car" else 1
    other_cells, other_direction = _projected_cells(world_matrix, agents, other_entity, other_step_count)
    other_current_cell = (int(other_position[0]), int(other_position[1]))
    overlap_cells = {cell for cell in ego_swept if cell in other_cells}
    overlap_outside_intersection = {
        cell for cell in overlap_cells if not intersection_mask[cell[0], cell[1]]
    }
    overlap_in_intersection = overlap_cells - overlap_outside_intersection
    same_direction = (
        agent_type == other_type == "Car"
        and direction is not None
        and other_direction == direction
    )

    if overlap_outside_intersection:
        if (not same_direction) or (other_current_cell in ego_swept):
            return 1

    other_in_intersection = bool(
        interior_intersection_mask[other_current_cell[0], other_current_cell[1]]
    )
    other_at_intersection = bool(
        _intersection_entry_mask(intersect_matrix, other_record.type)[
            other_current_cell[0], other_current_cell[1]
        ]
    )
    other_has_precedence = _has_strict_precedence(other_record, agent_record)

    if ego_in_intersection:
        if overlap_in_intersection and ((not same_direction) or other_in_intersection):
            return 1
        return 0

    if ego_at_intersection_after and other_in_intersection:
        return 1
    if ego_at_intersection_after and other_at_intersection and other_has_precedence:
        return 1

    if ego_in_intersection_after and overlap_in_intersection and ((not same_direction) or other_in_intersection):
        return 1

    return 0

def _safe_after_step_count(world_matrix, intersect_matrix, agents, entity, step_count):
    if "PH" in entity:
        return 0
    agent_type, layer_id, agent_layer, agent_position, agent_record = _get_entity_state(world_matrix, agents, entity)
    direction = _infer_moving_direction(agent_layer, agent_type, agent_position, agent_record.moving_direction)
    if direction is None:
        return 0

    ego_swept = _swept_cells(agent_position, direction, step_count, agent_layer.shape)
    if len(ego_swept) == 0:
        return 0

    intersection_mask = _intersection_mask(intersect_matrix)
    interior_intersection_mask = _intersection_interior_mask(intersect_matrix)
    entry_intersection_mask = _intersection_entry_mask(intersect_matrix, agent_type)
    ego_final_cell = ego_swept[-1]
    ego_hits_intersection = any(intersection_mask[x, y] for x, y in ego_swept)
    ego_in_intersection = bool(interior_intersection_mask[int(agent_position[0]), int(agent_position[1])])
    ego_at_intersection_after = bool(entry_intersection_mask[ego_final_cell[0], ego_final_cell[1]])
    ego_in_intersection_after = (
        bool(interior_intersection_mask[ego_final_cell[0], ego_final_cell[1]])
        or any(interior_intersection_mask[x, y] for x, y in ego_swept)
    )

    for other_key, other_record in agents.items():
        if other_key.startswith("ego_") and other_key == f"ego_{layer_id}":
            continue
        if not other_key.startswith("ego_") and str(other_record.layer_id) == layer_id:
            continue
        if other_record.type == "PH":
            continue

        other_entity = f"Entity_{other_record.type}_{other_record.layer_id}"
        other_step_count = 3 if other_record.type == "Car" else 1
        other_cells, other_direction = _projected_cells(world_matrix, agents, other_entity, other_step_count)
        other_current = _get_entity_state(world_matrix, agents, other_entity)[3]
        other_current_cell = (int(other_current[0]), int(other_current[1]))
        overlap_cells = {cell for cell in ego_swept if cell in other_cells}
        overlap_outside_intersection = {
            cell for cell in overlap_cells if not intersection_mask[cell[0], cell[1]]
        }
        overlap_in_intersection = overlap_cells - overlap_outside_intersection
        same_direction = (
            agent_type == other_record.type == "Car"
            and direction is not None
            and other_direction == direction
        )

        if overlap_outside_intersection:
            # Same-direction lane following should not be treated as an automatic
            # conflict just because both cars project through the same future cells.
            # We still block when the other agent already occupies a swept cell.
            if (not same_direction) or (other_current_cell in ego_swept):
                return 0

        other_in_intersection = bool(
            interior_intersection_mask[other_current_cell[0], other_current_cell[1]]
        )
        other_at_intersection = bool(
            _intersection_entry_mask(intersect_matrix, other_record.type)[
                other_current_cell[0], other_current_cell[1]
            ]
        )
        other_has_precedence = _has_strict_precedence(other_record, agent_record)

        if ego_in_intersection:
            if overlap_in_intersection and ((not same_direction) or other_in_intersection):
                return 0
            continue

        # Mirror the original easy rules:
        # - if ego is waiting at the intersection entry, wait for anyone already inside
        # - if ego is waiting at the intersection entry, wait for higher-priority agents
        #   that are also waiting at the intersection entry
        if ego_at_intersection_after and other_in_intersection:
            return 0
        if ego_at_intersection_after and other_at_intersection and other_has_precedence:
            return 0

        # If ego will already be inside after this action, only direct occupancy conflicts
        # should stop it; agents merely approaching the intersection should not.
        if ego_in_intersection_after and overlap_in_intersection and ((not same_direction) or other_in_intersection):
            return 0

    return 1

def SafeSlow(world_matrix, intersect_matrix, agents, entity):
    return _safe_after_step_count(world_matrix, intersect_matrix, agents, entity, 1)

def SafeNormal(world_matrix, intersect_matrix, agents, entity):
    return _safe_after_step_count(world_matrix, intersect_matrix, agents, entity, 2)

def SafeFast(world_matrix, intersect_matrix, agents, entity):
    return _safe_after_step_count(world_matrix, intersect_matrix, agents, entity, 3)

def SafeStop(world_matrix, intersect_matrix, agents, entity):
    if "PH" in entity:
        return 0
    return 1

def CollidingCloseStep1(world_matrix, intersect_matrix, agents, entity1, entity2):
    return _pairwise_step_conflict(world_matrix, intersect_matrix, agents, entity1, entity2, 1)

def CollidingCloseStep2(world_matrix, intersect_matrix, agents, entity1, entity2):
    return _pairwise_step_conflict(world_matrix, intersect_matrix, agents, entity1, entity2, 2)

def CollidingCloseStep3(world_matrix, intersect_matrix, agents, entity1, entity2):
    return _pairwise_step_conflict(world_matrix, intersect_matrix, agents, entity1, entity2, 3)

def IsSafeStep1(world_matrix, intersect_matrix, agents, entity):
    return SafeSlow(world_matrix, intersect_matrix, agents, entity)

def IsSafeStep2(world_matrix, intersect_matrix, agents, entity):
    return SafeNormal(world_matrix, intersect_matrix, agents, entity)

def IsSafeStep3(world_matrix, intersect_matrix, agents, entity):
    return SafeFast(world_matrix, intersect_matrix, agents, entity)

def IsSafeWait(world_matrix, intersect_matrix, agents, entity):
    return SafeStop(world_matrix, intersect_matrix, agents, entity)

def CollidingClose(world_matrix, intersect_matrix, agents, entity1, entity2):
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    _, agent_type1, layer_id1 = entity1.split("_")
    _, agent_type2, layer_id2 = entity2.split("_")
    if agent_type1 == "Pedestrian" or agent_type2 == "Pedestrian":
        return 0
    agent_layer1 = world_matrix[int(layer_id1)]
    agent_layer2 = world_matrix[int(layer_id2)]
    agent_position1 = (agent_layer1 == TYPE_MAP[agent_type1]).nonzero()[0]
    agent_position2 = (agent_layer2 == TYPE_MAP[agent_type2]).nonzero()[0]
    agent1_record = _get_agent_record_any(agents, layer_id1)
    agent1_dire = _infer_moving_direction(agent_layer1, agent_type1, agent_position1, _get_agent_direction(agent1_record))
    if agent1_dire is None:
        return 0
    return int(_in_forward_deadzone(agent_position1, agent_position2, agent1_dire))

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
