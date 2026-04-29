import re
import gc
import time
import copy
import torch
import logging
import importlib
import numpy as np
from z3 import *
from ...core.config import *
from multiprocessing import Pool
from ...utils.find import find_agent
from ...utils.sample import split_into_subsets
from .z3 import Z3Planner
from .z3 import PesudoAgent

logger = logging.getLogger(__name__)

class Z3PlannerRL(Z3Planner):
    def __init__(self, yaml_path):        
        super().__init__(yaml_path)
        self.rl_input_shape = None
        self.last_rl_obs = None

    def reset(self):
        self.last_rl_obs = None

    def _create_rules(self):
        assert "Task" in self.data["Rules"].keys(), "Make sure the task rule is defined"
        assert "Sim" in self.data["Rules"].keys(), "Make sure the sim rule is defined"
        self.rules = {
            "Task": {},
            "Sim": {}
        }
        for rule_dict in self.data["Rules"]["Sim"]:
            rule_name, formula = rule_dict["name"], rule_dict["formula"]
            # Check if the rule is valid
            logger.info("*** Sim Rule ***: {} -> \n {}".format(rule_name, formula))

            # Create Z3 variables based on the formula
            var_names = self._extract_variables(formula)
            # Note: Sim rule should contain all the Z3 variables

            # Substitute predicate names in the formula with Z3 function instances
            for method_name, pred_info in self.predicates.items():
                formula = formula.replace(method_name, f'local_predicates["{method_name}"]["instance"]')

            # Now replace the variable names in the formula with their Z3 counterparts
            for var_name in var_names:
                formula = formula.replace(var_name, f'z3_vars["{var_name}"]')
                if var_name not in self.z3_vars:
                    self.z3_vars.append(var_name)

            # Evaluate the modified formula string to create the Z3 expression
            self.rules["Sim"][rule_name] = formula

        for rule_dict in self.data["Rules"]["Task"]:
            rule_name, formula = rule_dict["name"], rule_dict["formula"]
            self.rules["Task"][rule_name] = {}
            # Check if the rule is valid
            logger.info("*** Task Rule ***: {} -> \n {}".format(rule_name, formula))

            # Create Z3 variables based on the formula
            var_names = self._extract_variables(formula)

            # Substitute predicate names in the formula with Z3 function instances
            for method_name, pred_info in self.predicates.items():
                formula = formula.replace(method_name, f'local_predicates["{method_name}"]["instance"]')

            # Now replace the variable names in the formula with their Z3 counterparts
            for var_name in var_names:
                formula = formula.replace(var_name, f'z3_vars["{var_name}"]')

            # Evaluate the modified formula string to create the Z3 expression
            self.rules["Task"][rule_name]["content"] = formula
            self.rules["Task"][rule_name]["reward"] = rule_dict["reward"]
            self.rules["Task"][rule_name]["dead"] = rule_dict["dead"]
        
        logger.info("Rules created successfully")
        logger.info("Rules will be grounded later...")

    def plan(self, world_matrix, 
             intersect_matrix, 
             agents, 
             layerid2listid, 
             use_multiprocessing=True, 
             rl_agent=None):
        # 1. Break the global world matrix into local world matrix and split the agents and intersections
        # Note that the local ones will have different size and agent id
        # e = time.time()
        local_world_matrix = world_matrix.clone()
        local_intersections = intersect_matrix.clone()
        ego_agent, partial_agents, partial_world, partial_intersections, rl_flags = \
            self.break_world_matrix(local_world_matrix, agents, local_intersections, layerid2listid, rl_agent)
        # logger.info("Break world time: {}".format(time.time()-e))
        # 2. Choose between multi-processing and looping
        combined_results = {}
        agent_keys = list(partial_agents.keys())
        
        if use_multiprocessing:
            # Multi-processing approach
            agent_batches = split_into_batches(agent_keys, NUM_PROCESS)
            with Pool(processes=NUM_PROCESS) as pool:
                for batch_keys in agent_batches:
                    batch_results = pool.starmap(solve_sub_problem, 
                                                [(ego_name, ego_agent[ego_name].action_mapping, ego_agent[ego_name].action_dist,
                                                self.rule_tem, self.entity_types, self.predicates, self.z3_vars,
                                                partial_agents[ego_name], partial_world[ego_name], partial_intersections[ego_name],
                                                self.fov_entities, rl_flags[ego_name], self.rl_input_shape)
                                                for ego_name in batch_keys])
                    
                    for result in batch_results:
                        combined_results.update(result)
                    gc.collect()
        else:
            # Looping approach
            for ego_name in agent_keys:
                if rl_flags[ego_name]:
                    # RL agent only gets the observation
                    result = solve_sub_problem(ego_name, ego_agent[ego_name].action_mapping, ego_agent[ego_name].action_dist,
                                            self.rules['Task'], self.entity_types, self.predicates, self.z3_vars,
                                            partial_agents[ego_name], partial_world[ego_name], partial_intersections[ego_name], 
                                            self.fov_entities, True, rl_input_shape=self.rl_input_shape)
                    self.last_rl_obs = {
                        "last_obs_dict": copy.deepcopy(result["{}_grounding_dic".format(ego_name)]),
                        "last_obs": result["{}_grounding".format(ego_name)].copy(),
                        "shield_context": copy.deepcopy(result.get("{}_shield_context".format(ego_name))),
                    }
                else:
                    result = solve_sub_problem(ego_name, ego_agent[ego_name].action_mapping, ego_agent[ego_name].action_dist,
                                            self.rules['Sim'], self.entity_types, self.predicates, self.z3_vars,
                                            partial_agents[ego_name], partial_world[ego_name], partial_intersections[ego_name], 
                                            self.fov_entities, False)
                combined_results.update(result)

        e2 = time.time()
        # logger.info("Solve sub-problem time: {}".format(e2-e))
        return combined_results
    
    def eval(self, rl_action):
        if self.last_rl_obs is None:
            return 0
        fail, reward = eval_action(rl_action, self.rules['Task'], self.entity_types, self.predicates, self.z3_vars, self.fov_entities,
                             self.last_rl_obs["last_obs_dict"], self.last_rl_obs["last_obs"])
        self.last_rl_obs = None
        return fail, reward

    def break_world_matrix(self, world_matrix, agents, intersect_matrix, layerid2listid, rl_agent):
        ego_agent = {}
        partial_agents = {}
        partial_world = {}
        partial_intersection = {}
        rl_flag = {}
        for agent in agents:
            ego_name = "{}_{}".format(agent.type, agent.layer_id)
            ego_agent[ego_name] = agent
            rl_flag[ego_name] = (agent.layer_id==rl_agent)
            ego_layer = world_matrix[agent.layer_id]
            assert len((ego_layer == TYPE_MAP[agent.type]).nonzero()) == 1, ValueError("Ego agent {}_{} should be unique in the world matrix, now it is {}".format(agent.type, agent.layer_id, (ego_layer == TYPE_MAP[agent.type]).nonzero()))
            ego_position = (ego_layer == TYPE_MAP[agent.type]).nonzero()[0]
            ego_direction = agent.last_move_dir
            x_start, y_start, x_end, y_end = self.get_fov(ego_position, ego_direction, world_matrix.shape[1], world_matrix.shape[2])
            partial_world_all = world_matrix[:, x_start:x_end, y_start:y_end].clone()
            partial_intersections = intersect_matrix[:, x_start:x_end, y_start:y_end].clone()
            partial_world_nonzero_int = torch.logical_and(partial_world_all != 0, \
                                                          partial_world_all == partial_world_all.to(torch.int64))
            # Apply torch.any across dimensions 1 and 2 sequentially
            non_zero_layers = partial_world_nonzero_int.any(dim=1).any(dim=1)
            non_zero_layer_indices = torch.where(non_zero_layers)[0]
            partial_world_squeezed = partial_world_all[non_zero_layers]
            partial_agent = {}
            for layer_id in range(partial_world_squeezed.shape[0]):
                layer = partial_world_squeezed[layer_id]
                layer_nonzero_int = torch.logical_and(layer != 0, layer == layer.to(torch.int64))
                if layer_nonzero_int.nonzero().shape[0] > 1:
                    continue
                if len(partial_agent) >= self.fov_entities["Entity"] and rl_flag[ego_name]:
                    # can only handle fixed number of agents
                    break
                non_zero_values = int(layer[layer_nonzero_int.nonzero()[0][0], layer_nonzero_int.nonzero()[0][1]])
                agent_type = LABEL_MAP[non_zero_values]
                # find this agent
                other_agent_layer_id = int(non_zero_layer_indices[layer_id])
                other_agent = agents[layerid2listid[other_agent_layer_id]]
                assert other_agent.type == agent_type
                if other_agent_layer_id == agent.layer_id:
                    partial_agent["ego_{}".format(layer_id)] = PesudoAgent(agent_type, layer_id, other_agent.concepts, other_agent.last_move_dir)
                else:
                    partial_agent[str(layer_id)] = PesudoAgent(agent_type, layer_id, other_agent.concepts, other_agent.last_move_dir)
            if rl_flag[ego_name]:
                # RL agent needs fixed number of entities
                ph_concepts = {
                    "type": "PH",
                    "priority": 0,
                }
                while len(partial_agent) < self.fov_entities["Entity"]:
                    layer_id += 1
                    place_holder_agent = PesudoAgent("PH", layer_id, ph_concepts, \
                                                      None)
                    partial_agent["PH_{}".format(layer_id)] = place_holder_agent
                        # Additional place holder for rl agent
                    # partial_world_squeezed = torch.cat([partial_world_squeezed, partial_world_squeezed[-1].unsqueeze(0)], dim=0)
            partial_world[ego_name] = partial_world_squeezed
            partial_intersection[ego_name] = partial_intersections
            partial_agents[ego_name] = partial_agent
        return ego_agent, partial_agents, partial_world, partial_intersection, rl_flag
            
    def logic_grounding_shape(self, fov_entities):
        self.fov_entities = fov_entities
        self.rl_input_shape, pred_grounding_ind = logic_grounding_shape(self.entity_types, self.predicates, self.z3_vars, fov_entities)
        self.pred_grounding_index = pred_grounding_ind
        return self.rl_input_shape, pred_grounding_ind

def logic_grounding_shape(
                      entity_types, 
                      predicates, 
                      var_names,
                      fov_entities):
    # TODO: determine the shape of the logic grounding in the RL agent
    n = 0
    pred_grounding_index = {}
    # 1. create sorts and variables
    entity_sorts = {}
    for entity_type in entity_types:
        entity_sorts[entity_type] = DeclareSort(entity_type)
        assert fov_entities[entity_type] > 0, "Make sure the entity type (defined in rules) is in the fov_entities"
    # 3. entities
    entities = {}
    for entity_type in entity_sorts.keys():
        entity_num = fov_entities[entity_type]
        entities[entity_type] = [Const(f"{entity_type}_{i}", entity_sorts[entity_type]) for i in range(entity_num)]
    # 4. create, ground predicates and add to solver
    local_predicates = copy.deepcopy(predicates)
    for pred_name, pred_info in local_predicates.items():
        eval_pred = eval(pred_info["instance"])
        pred_info["instance"] = eval_pred
        arity = pred_info["arity"]

        method_full_name = pred_info["function"]
        if method_full_name == "None":
            continue

        if arity == 1:
            n_start = n
            # Unary predicate grounding
            for _ in entities[eval_pred.domain(0).name()]:
                n += 1
            pred_grounding_index[pred_name] = (n_start, n)
        elif arity == 2:
            n_start = n
            # Binary predicate grounding
            for _ in entities[eval_pred.domain(0).name()]:
                for _ in entities[eval_pred.domain(1).name()]:
                    n += 1
            pred_grounding_index[pred_name] = (n_start, n)
    logger.info("Given Predicates {}, the FOV entities {}, The logic grounding shape is: {}".format(local_predicates, fov_entities, n))
    return n, pred_grounding_index

def solve_sub_problem(ego_name, 
                      ego_action_mapping,
                      ego_action_dist,
                      rule_tem, 
                      entity_types, 
                      predicates, 
                      var_names,
                      partial_agents, 
                      partial_world, 
                      partial_intersections,
                      fov_entities,
                      rl_flag,
                      rl_input_shape=None):
    grounding = []
    grounding_dic = {}
    # 1. create sorts and variables
    entity_sorts = {}
    for entity_type in entity_types:
        entity_sorts[entity_type] = DeclareSort(entity_type)
    z3_vars = {var_name: Const(var_name, entity_sorts["Entity"]) \
                       for var_name in var_names}
    # 2. partial world to entities
    local_entities = world2entity(entity_sorts, partial_intersections, partial_agents, fov_entities, rl_flag)
    # 3. create, ground predicates and add to solver
    local_predicates = copy.deepcopy(predicates)
    # 4. create, ground predicates and add to solver
    if not rl_flag:
        local_solver = Solver()
        for pred_name, pred_info in local_predicates.items():
            eval_pred = eval(pred_info["instance"])
            pred_info["instance"] = eval_pred
            arity = pred_info["arity"]

            # Import the grounding method
            method_full_name = pred_info["function"]
            if method_full_name == "None":
                continue
            module_name, method_name = method_full_name.rsplit('.', 1)
            module = importlib.import_module(module_name)
            method = getattr(module, method_name)

            if arity == 1:
                # Unary predicate grounding
                for entity in local_entities[eval_pred.domain(0).name()]:
                    entity_name = entity.decl().name()
                    value = method(partial_world, partial_intersections, partial_agents, entity_name)
                    if value:
                        local_solver.add(eval_pred(entity))
                    else:
                        local_solver.add(Not(eval_pred(entity)))
            elif arity == 2:
                # Binary predicate grounding
                for entity1 in local_entities[eval_pred.domain(0).name()]:
                    entity1_name = entity1.decl().name()
                    for entity2 in local_entities[eval_pred.domain(1).name()]:
                        entity2_name = entity2.decl().name()
                        value = method(partial_world, partial_intersections, partial_agents, entity1_name, entity2_name)
                        if value:
                            local_solver.add(eval_pred(entity1, entity2))
                        else:
                            local_solver.add(Not(eval_pred(entity1, entity2)))
        # 5. create, ground rules and add to solver
        local_rule_tem = copy.deepcopy(rule_tem)
        for rule_name, rule_template in local_rule_tem.items():
            # the first entity is the ego agent
            entity = local_entities["Entity"][0]
            # Replace placeholder in the rule template with the actual agent entity
            instantiated_rule = eval(rule_template)
            local_solver.add(instantiated_rule)

        # **Important: Closed world quantifier rule, to ensure z3 do not add new entity to satisfy the rule and "dummy" is not part of the world**
        for var_name, z3_var in z3_vars.items():
            entity_list = local_entities["Entity"]
            constraint = Or([z3_var == entity for entity in entity_list])
            local_solver.add(ForAll([z3_var], constraint))
        
        # 6. solve
        if local_solver.check() == sat:
            model = local_solver.model()
            # Interpret the solution to the FOL problem
            action_mapping = ego_action_mapping
            action_dist = torch.zeros_like(ego_action_dist)

            for key in local_predicates.keys():
                action = []
                for action_id, action_name in action_mapping.items():
                    if key in action_name:
                        action.append(action_id)
                if len(action)>0:
                    for a in action:
                        if is_true(model.evaluate(local_predicates[key]["instance"](local_entities["Entity"][0]))):
                            action_dist[a] = 1.0
            # No action specified, use the default action, Normal
            if action_dist.sum() == 0:
                for action_id, action_name in action_mapping.items():
                    if "Normal" in action_name:
                        action_dist[action_id] = 1.0

            agents_actions = {ego_name: action_dist}
            return agents_actions
        else:
            action_mapping = ego_action_mapping
            action_dist = torch.zeros_like(ego_action_dist)

            for action_id, action_name in action_mapping.items():
                if "Normal" in action_name:
                    action_dist[action_id] = 1.0

            agents_actions = {ego_name: action_dist}
            return agents_actions
    else:
        for pred_name, pred_info in local_predicates.items():
            k = 0
            eval_pred = eval(pred_info["instance"])
            pred_info["instance"] = eval_pred
            arity = pred_info["arity"]

            # Import the grounding method
            method_full_name = pred_info["function"]

            if method_full_name == "None":
                continue

            module_name, method_name = method_full_name.rsplit('.', 1)
            module = importlib.import_module(module_name)
            method = getattr(module, method_name)

            if arity == 1:
                # Unary predicate grounding
                for entity in local_entities[eval_pred.domain(0).name()]:
                    entity_name = entity.decl().name()
                    value = method(partial_world, partial_intersections, partial_agents, entity_name)
                    if value:
                        grounding_dic["{}_{}".format(pred_name, k)] = 1
                        grounding.append(1)
                        k += 1
                    else:
                        grounding_dic["{}_{}".format(pred_name, k)] = 0
                        grounding.append(0)
                        k += 1
            elif arity == 2:
                # Binary predicate grounding
                for entity1 in local_entities[eval_pred.domain(0).name()]:
                    entity1_name = entity1.decl().name()
                    for entity2 in local_entities[eval_pred.domain(1).name()]:
                        entity2_name = entity2.decl().name()
                        value = method(partial_world, partial_intersections, partial_agents, entity1_name, entity2_name)
                        if value:
                            grounding_dic["{}_{}".format(pred_name, k)] = 1
                            grounding.append(1)
                            k += 1
                        else:
                            grounding_dic["{}_{}".format(pred_name, k)] = 0
                            grounding.append(0)
                            k += 1

        agents_actions = {
            "{}_grounding".format(ego_name): np.array(grounding, dtype=np.float32),
            "{}_grounding_dic".format(ego_name): grounding_dic
        }
        ego_agent = None
        for key, agent in partial_agents.items():
            if "ego" in key:
                ego_agent = agent
                break
        if ego_agent is not None:
            agents_actions["{}_shield_context".format(ego_name)] = {
                "world_matrix": partial_world.clone(),
                "intersect_matrix": partial_intersections.clone(),
                "agents": copy.deepcopy(partial_agents),
                "ego_entity": f"Entity_{ego_agent.type}_{ego_agent.layer_id}",
            }
        assert len(grounding) == rl_input_shape

        return agents_actions

def eval_action(rl_action,
                rule_tem, 
                entity_types, 
                predicates, 
                var_names,
                fov_entities,
                last_obs_dict,
                last_obs):
    grounding = []
    # 1. create sorts and variables
    entity_sorts = {}
    for entity_type in entity_types:
        entity_sorts[entity_type] = DeclareSort(entity_type)
    z3_vars = {var_name: Const(var_name, entity_sorts["Entity"]) \
                       for var_name in var_names}
    # 2. entities
    entities = {}
    for entity_type in entity_sorts.keys():
        entity_num = fov_entities[entity_type]
        entities[entity_type] = [Const(f"{entity_type}_{i}", entity_sorts[entity_type]) for i in range(entity_num)]
    # 3. create, ground predicates and add to solver
    local_predicates = copy.deepcopy(predicates)
    # 4. create, ground predicates and add to solver
    local_solvers = {rule_name: Solver() for rule_name in rule_tem.keys()}
    for pred_name, pred_info in local_predicates.items():
        k = 0
        eval_pred = eval(pred_info["instance"])
        pred_info["instance"] = eval_pred
        arity = pred_info["arity"]

        # Import the grounding method
        method_full_name = pred_info["function"]

        if method_full_name == "None":
            assert rl_action is not None, "Make sure the rl_action is not None"
            # ego action
            action_name = get_action_name(rl_action)
            if pred_name == action_name:
                for rule_name, rule_template in rule_tem.items():
                    local_solvers[rule_name].add(eval_pred(entities["Entity"][0]))
            else:
                for rule_name, rule_template in rule_tem.items():
                    local_solvers[rule_name].add(Not(eval_pred(entities["Entity"][0])))
            continue

        if arity == 1:
            # Unary predicate grounding
            for entity in entities[eval_pred.domain(0).name()]:
                if last_obs_dict["{}_{}".format(pred_name, k)]:
                    grounding.append(1)
                    k += 1
                    for rule_name, rule_template in rule_tem.items():
                        local_solvers[rule_name].add(eval_pred(entity))
                else:
                    grounding.append(0)
                    k += 1
                    for rule_name, rule_template in rule_tem.items():
                        local_solvers[rule_name].add(Not(eval_pred(entity)))
        elif arity == 2:
            # Binary predicate grounding
            for entity1 in entities[eval_pred.domain(0).name()]:
                for entity2 in entities[eval_pred.domain(1).name()]:
                    if last_obs_dict["{}_{}".format(pred_name, k)]:
                        grounding.append(1)
                        k += 1
                        for rule_name, rule_template in rule_tem.items():
                            local_solvers[rule_name].add(eval_pred(entity1, entity2))
                    else:
                        grounding.append(0)
                        k += 1
                        for rule_name, rule_template in rule_tem.items():
                            local_solvers[rule_name].add(Not(eval_pred(entity1, entity2)))

    # 5. create, ground rules and add to solver
    local_rule_tem = copy.deepcopy(rule_tem)
    for rule_name, rule_template in local_rule_tem.items():
        # the first entity is the ego agent
        entity = entities["Entity"][0]
        # Replace placeholder in the rule template with the actual agent entity
        instantiated_rule = eval(rule_template["content"])
        local_solvers[rule_name].add(instantiated_rule)

    # **Important: Closed world quantifier rule, to ensure z3 do not add new entity to satisfy the rule and "dummy" is not part of the world**
    for var_name, z3_var in z3_vars.items():
        entity_list = entities['Entity']
        for rule_name, rule_template in rule_tem.items():
            constraint = Or([z3_var == entity for entity in entity_list])
            local_solvers[rule_name].add(ForAll([z3_var], constraint))
    
    # 6. solve for reward
    obs = np.array(grounding, dtype=np.float32)
    assert np.all(obs == last_obs), print(obs, last_obs)
    fail = False
    reward = 0
    for rule_name, rule_solver in local_solvers.items():
        if rule_solver.check() == sat:
                continue
        else:
            if rule_tem[rule_name]["dead"]:
                fail = True
            reward += local_rule_tem[rule_name]["reward"]

    return fail, reward

def get_action_name(rl_action):
    # see agents/car.py
    if rl_action[0] == 1:
        return "Slow"
    elif rl_action[4] == 1:
        return "Normal"
    elif rl_action[8] == 1:
        return "Fast"
    else:
        return "Stop"

def split_into_batches(keys, batch_size):
    """Split keys into batches of a given size."""
    for i in range(0, len(keys), batch_size):
        yield keys[i:i + batch_size]

def world2entity(entity_sorts, partial_intersect, partial_agents, fov_entities, rl_flag):
    # all the enitities are stored in self.entities
    entities = {}
    for entity_type in entity_sorts.keys():
        entities[entity_type] = []
        flag = False
        # For Agents
        if entity_type == "Entity":
            for key, agent in partial_agents.items():
                if "ego" in key:
                    ego_agent = agent
                    flag = True
                    continue
                if "PH" in key:
                    agent_id = agent.layer_id
                    agent_name = f"Entity_PH_{agent_id}"
                else:
                    agent_id = agent.layer_id
                    agent_type = agent.type
                    agent_name = f"Entity_{agent_type}_{agent_id}"
                # Create a Z3 constant for the agent
                agent_entity = Const(agent_name, entity_sorts['Entity'])
                entities[entity_type].append(agent_entity)
            assert flag, logger.info(partial_agents)
            agent_id = ego_agent.layer_id
            agent_type = ego_agent.type
            agent_name = f"Entity_{agent_type}_{agent_id}"
            # Create a Z3 constant for the agent
            agent_entity = Const(agent_name, entity_sorts['Entity'])
            # ego agent is the first
            entities[entity_type] = [agent_entity] + entities[entity_type]
            if rl_flag:
                assert len(entities[entity_type]) == fov_entities["Entity"], logger.info(entities)
    return entities
