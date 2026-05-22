from lnn import *
from .basic import LocalPlanner
from ...core.config import *
from ...utils.find import find_entity
from ...utils.check import check_fol_rule_syntax
import importlib
import numpy as np
import torch

class LNNPlanner(LocalPlanner):
    def __init__(self, yaml_path):        
        super().__init__(yaml_path)
        
    def _create_predicates(self):
        # Using a dictionary to store the arity as well
        self.predicates = {}
        for p in self.data["predicates"]:
            predicate, info = list(p.items())[0]
            self.predicates[predicate] = {
                "instance": Predicate(predicate, info["arity"]),
                "method": info["method"],
                "arity": info["arity"],
                "description": info["description"]
            }
        
    def _create_rules(self):
        # For the sake of simplicity, we'll only handle basic logical constructs here (And, Or, Implies)
        # More complex rules might require further adjustments
        logical_mapping = {
            'And': And,
            'Or': Or,
            'Implies': Implies,
            'Not': Not,
            'Exists': Exists,
            'Forall': Forall
        }
        self.model_dict = {}
        self.rule_dict = {}
        self.model_preds = {}
        x, y = Variables('x', 'y') # maximum 2 variables for now
        
        for r in self.data["rules"]:
            local_model = Model()
            rule_name, rule_info = list(r.items())[0]
            formula_str = rule_info["formula"]
            check_fol_rule_syntax(formula_str)
            
            # Replace formula's string content to make it Python executable
            for predicate in self.predicates:
                formula_str = formula_str.replace(predicate, "self.predicates['{}']['instance']".format(str(predicate)))
            
            for key, func in logical_mapping.items():
                formula_str = formula_str.replace(key, func.__name__)
            
            rule_instance = eval(formula_str)  # This dynamically evaluates the Python equivalent formula
            self.rule_dict[rule_name] = rule_instance
            # All the rules are considered as axioms for now
            local_model.add_knowledge(rule_instance, world=World.AXIOM)
            self.model_dict[rule_name] = local_model
            model_pred = []
            for key in local_model.nodes.keys():
                model_pred.append(local_model.nodes[key])
            self.model_preds[rule_name] = model_pred
    
    def _create_entities(self):
        # Create the entities
        self.entity_types = self.data["entities"]
        self.entity_list = []
    
    # Process world matrix to ground the world state predicates
    def add_world_data(self, world_matrix, intersect_matrix, agents):
        # 1. Convert the world to predicates, symbolic grounding
        data_dict = {}

        for p in self.predicates.keys():
            predicate_info = self.predicates[p]
            arity = predicate_info["arity"]
            data_dict[predicate_info["instance"]] = {}

            method_full_name = predicate_info["method"]
            if method_full_name == "None":
                # Means this predicate is not grounded by the world, it is an action predicate
                continue
            module_name, method_name = method_full_name.rsplit('.', 1)
            module = importlib.import_module(module_name)
            method = getattr(module, method_name)

            if arity == 1:
                # Unary predicate processing
                for entity in self.entity_list:
                    values = method(world_matrix, intersect_matrix, agents, entity)
                    data_dict[predicate_info["instance"]][entity] = values
            elif arity == 2:
                # Binary predicate processing
                for entity1 in self.entity_list:
                    for entity2 in self.entity_list:
                        values = method(world_matrix, intersect_matrix, agents, entity1, entity2)
                        data_dict[predicate_info["instance"]][(entity1, entity2)] = values
        for rule_name, model in self.model_dict.items():
            # 2. Add the grounded predicates as facts to the model
            model_dict = {}
            for key in data_dict.keys():
                if key in self.model_preds[rule_name]:
                    model_dict[key] = data_dict[key]
            model.add_data(model_dict)
            # 3. Add the rule as known truth to the model
            model.add_knowledge(self.rule_dict[rule_name], world=World.AXIOM)


    def plan(self, world_matrix, intersect_matrix, agents):
        for model in self.model_dict.values():
            model.flush()
        # Add the world data to the predicates, this may need enumerate all groundings
        if len(self.entity_list) == 0:
            # initialize the entity list
            self.world2entity(world_matrix, intersect_matrix, agents)
        self.add_world_data(world_matrix, intersect_matrix, agents)
        for model in self.model_dict.values():
            model.infer(Direction.UPWARD)
            model.infer(Direction.DOWNWARD)
        # use LNN to get the action distribution
        agents_actions = {}
        for agent in agents:
            entity_name = find_entity(agent)
            agent_name = "{}_{}".format(agent.type, agent.layer_id)
            action_mapping = agent.action_mapping
            action_dist = torch.zeros_like(agent.action_dist)
            for key in self.predicates.keys():
                action = []
                for action_id, action_name in action_mapping.items():
                    if key in action_name:
                        action.append(action_id)
                if len(action)>0:
                    for a in action:
                        action_dist[a] = self.convert(key, entity_name)
            # No action specified, use the default action, Normal
            if action_dist.sum() == 0:
                for action_id, action_name in action_mapping.items():
                    if "Normal" in action_name:
                        action_dist[action_id] = 1.0
            agents_actions[agent_name] = action_dist
        return agents_actions

    def convert(self, key_name, agent_name):
        pred_list = []
        pred_list.append(self.predicates[key_name]["instance"].get_data(agent_name))
        LU_bound = torch.cat(pred_list, dim=0)

        value = torch.avg_pool1d(LU_bound, kernel_size=2)
        value[value==0.5] = 0.0
        value = torch.clip(value.sum(), 0.0, 1.0)
        return value
    
    def world2entity(self, world_matrix, intersect_matrix, agents):
        for entity_type in self.entity_types:
            if entity_type == 'Agents':
                entity_name = "Agents_{}_{}"
                for agent in agents:
                    agent_name = find_entity(agent)
                    self.entity_list.append(agent_name)
            elif entity_type == 'Intersections':
                unique_intersections = np.unique(intersect_matrix[0])
                unique_intersections = unique_intersections[unique_intersections != 0]
                for i in unique_intersections:
                    entity_name = "{}_{}".format(entity_type, i)
                    self.entity_list.append(entity_name)
                assert len(unique_intersections) > 0
