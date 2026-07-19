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
    # intersection, not only a narrow precomputed line mask. In the compact
    # 2x2 benchmark, cars often sit one cell off the line mask while still
    # being the next vehicle to enter the same intersection. We therefore
    # treat a car as "at" the intersection when it is currently outside the
    # block but has an adjacent route point inside that block.
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
        neighbor_points = []
        if current_idx > 0:
            neighbor_points.append(traj[current_idx - 1])
        if current_idx < len(traj) - 1:
            neighbor_points.append(traj[current_idx + 1])

        for point in neighbor_points:
            if intersect_matrix[2, point[0], point[1]]:
                return 1
        if intersect_matrix[0, agent_position[0], agent_position[1]]:
            return 1
        return 0
    else:
        if intersect_matrix[1, agent_position[0], agent_position[1]]:
            return 1
        else:
            return 0


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

    if agent_type != "Car":
        return 0

    current_line_inter_id = int(intersect_matrix[0, agent_position[0], agent_position[1]].item())
    if current_line_inter_id > 0:
        return current_line_inter_id

    agent_key = str(layer_id) if str(layer_id) in agents else "ego_{}".format(layer_id)
    if agent_key not in agents:
        return 0
    agent = agents[agent_key]
    traj = getattr(agent, "global_traj", None)
    if traj is None or len(traj) == 0:
        return 0

    matches = torch.all(traj == agent_position, dim=1).nonzero(as_tuple=False)
    if matches.numel() == 0:
        return 0

    current_idx = int(matches[0].item())

    # Prefer the upcoming intersection along the agent's route. Falling back to
    # a previous route point can incorrectly bind a car to the intersection it
    # has just left, which delays SameInter for approaching traffic.
    lookahead_limit = min(len(traj), current_idx + 7)
    for idx in range(current_idx + 1, lookahead_limit):
        point = traj[idx]
        inter_id = int(intersect_matrix[2, point[0], point[1]].item())
        if inter_id > 0:
            return inter_id
        line_inter_id = int(intersect_matrix[0, point[0], point[1]].item())
        if line_inter_id > 0:
            return line_inter_id

    lookbehind_start = max(0, current_idx - 3)
    for idx in range(current_idx - 1, lookbehind_start - 1, -1):
        point = traj[idx]
        inter_id = int(intersect_matrix[2, point[0], point[1]].item())
        if inter_id > 0:
            return inter_id
        line_inter_id = int(intersect_matrix[0, point[0], point[1]].item())
        if line_inter_id > 0:
            return line_inter_id
    return 0


def _get_agent_direction(agents, layer_id):
    agent = None
    if layer_id in agents.keys():
        agent = agents[layer_id]
    else:
        ego_key = "ego_{}".format(layer_id)
        assert ego_key in agents.keys()
        agent = agents[ego_key]
    if hasattr(agent, "moving_direction"):
        return agent.moving_direction
    return getattr(agent, "last_move_dir", None)

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
    # TODO: Colliding close checker
    # 1. Get the position of the two agents
    _, agent_type1, layer_id1 = entity1.split("_")
    _, agent_type2, layer_id2 = entity2.split("_")
    agent_layer1 = world_matrix[int(layer_id1)]
    agent_layer2 = world_matrix[int(layer_id2)]
    agent_position1 = (agent_layer1 == TYPE_MAP[agent_type1]).nonzero()[0]
    agent_position2 = (agent_layer2 == TYPE_MAP[agent_type2]).nonzero()[0]
    # 2. Get the moving direction of the first agent
    agent1_dire = _get_agent_direction(agents, layer_id1)
    agent2_dire = _get_agent_direction(agents, layer_id2)
    if agent1_dire == None:
        return 0
    else:
        if agent_type1 == "Car" and agent_type2 == "Car":
            same_inter = SameInter(world_matrix, intersect_matrix, agents, entity1, entity2)
            if same_inter:
                ego_at_inter = IsAtInter(world_matrix, intersect_matrix, agents, entity1)
                other_at_inter = IsAtInter(world_matrix, intersect_matrix, agents, entity2)
                ego_in_inter = IsInInter(world_matrix, intersect_matrix, agents, entity1)
                other_in_inter = IsInInter(world_matrix, intersect_matrix, agents, entity2)
                # In dense validation scenes, collision fallback was triggering
                # too early for same-intersection traffic, masking the intended
                # right-of-way rules. Keep the fallback only once interaction is
                # truly imminent: one car already inside, or both queued at the
                # intersection boundary.
                if not (ego_in_inter or other_in_inter or (ego_at_inter and other_at_inter)):
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
