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

def _get_agent_record(agents, layer_id):
    if layer_id in agents:
        return agents[layer_id]
    ego_key = f"ego_{layer_id}"
    if ego_key in agents:
        return agents[ego_key]
    raise KeyError(f"Agent layer {layer_id} not found in partial agent map.")

def _get_entity_state(world_matrix, agents, entity):
    _, agent_type, layer_id = entity.split("_")
    agent_layer = world_matrix[int(layer_id)]
    agent_position = (agent_layer == TYPE_MAP[agent_type]).nonzero()[0]
    agent_record = _get_agent_record(agents, layer_id)
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
    direction = _infer_moving_direction(agent_layer, agent_type, agent_position, agent_record.moving_direction)
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
    # TODO: Colliding close checker
    # 1. Get the position of the two agents
    _, agent_type1, layer_id1 = entity1.split("_")
    _, agent_type2, layer_id2 = entity2.split("_")
    agent_layer1 = world_matrix[int(layer_id1)]
    agent_layer2 = world_matrix[int(layer_id2)]
    agent_position1 = (agent_layer1 == TYPE_MAP[agent_type1]).nonzero()[0]
    agent_position2 = (agent_layer2 == TYPE_MAP[agent_type2]).nonzero()[0]
    # 2. Get the moving direction of the first agent
    if layer_id1 in agents.keys():
        agent1_dire = agents[layer_id1].moving_direction
    else:
        assert "ego_{}".format(layer_id1) in agents.keys()
        agent1_dire = agents["ego_{}".format(layer_id1)].moving_direction
    if agent1_dire == None:
        return 0
    else:
        dist = torch.sqrt(torch.sum((agent_position1 - agent_position2)**2))
        if dist > OCC_CHECK_RANGE[agent_type1]:
            return 0
        elif dist == 0:
            return np.random.choice([0, 1], p=[0.5, 0.5])
        else:
            agent1_dire_vec = torch.tensor(DIRECTION_VECTOR[agent1_dire])
            angle = torch.acos(torch.dot(agent1_dire_vec, (agent_position2 - agent_position1)) / dist)
            if angle < OCC_CHECK_ANGEL:
                if agent_type1 == "Car":
                    # Cars will definitely stop to avoid collide
                    return 1
                else:
                    # Pedestrians will probably stop to avoid collide
                    sample = np.random.rand()
                    if sample < PED_AGGR:
                        return 1
                    else:
                        return 0
    return 0

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
