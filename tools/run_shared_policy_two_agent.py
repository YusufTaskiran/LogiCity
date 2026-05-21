import os
import copy
import csv
import json
import time
import yaml
import torch
import argparse
import importlib
import numpy as np
import pickle as pkl
from collections import Counter

from logicity.utils.load import CityLoader
from logicity.utils.logger import setup_logger
from logicity.utils.pred_converter.z3 import global_hard_failure_breakdown, global_stop_required_conflict
from logicity.shields import JointProbabilisticLogicShieldTwoCar
from stable_baselines3.common.utils import obs_as_tensor


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run one shared PPO policy on two RL-controlled agents.")
    parser.add_argument("--config", default="config/tasks/Nav/simple/algo/ppo_two_agent_shared_infer.yaml")
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--episode_data", default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--log_dir", type=str, default="./log_rl")
    parser.add_argument("--exp", type=str, default="simple_two_agent_shared")
    parser.add_argument("--save_worlds", action="store_true")
    parser.add_argument("--seed", type=int, default=2)
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def dynamic_import(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_city(simulation_config, episode_cache=None):
    city, cached_observation = CityLoader.from_yaml(**simulation_config, episode_cache=episode_cache)
    return city, cached_observation


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class SharedPolicyTwoAgentRunner:
    def __init__(self, city, rl_agent_cfg, controlled_agent_names):
        self.city = city
        self.rl_agent_cfg = rl_agent_cfg
        self.controlled_agent_names = list(controlled_agent_names)
        self.cat_length = rl_agent_cfg.get("cat_length", False)
        self.grounding_mode = rl_agent_cfg.get("grounding_mode")
        self.action_mapping = rl_agent_cfg["action_mapping"]
        self.action_cost = rl_agent_cfg["action_cost"]
        self.num_actions = int(rl_agent_cfg["action_space"])
        self.stop_action_id = int(rl_agent_cfg.get("stop_action_id", self.num_actions - 1))
        self.horizon = rl_agent_cfg["max_horizon"]
        self.overtime_cost = rl_agent_cfg.get("overtime_cost", -3)
        self.controlled_agents = self._resolve_controlled_agents()
        self.controlled_layer_ids = {agent.layer_id for agent in self.controlled_agents}
        self.path_lengths = {
            agent.layer_id: max(len(agent.global_traj) * 4, 1) for agent in self.controlled_agents
        }
        self.normed_path_lengths = {
            agent.layer_id: len(agent.global_traj) / (2 * agent.region) for agent in self.controlled_agents
        }
        self.layer_id_to_name = {agent.layer_id: f"{agent.type}_{agent.id}" for agent in self.controlled_agents}
        self.layer_id_to_priority = {agent.layer_id: int(agent.priority) for agent in self.controlled_agents}
        self.t = 0
        self.parked_controlled_layers = set()

    def _entity_name_from_layer(self, layer_id):
        for agent in self.city.agents:
            if int(agent.layer_id) == int(layer_id):
                return f"Entity_{agent.type}_{agent.layer_id}"
        raise KeyError(f"Agent layer {layer_id} not found.")

    def _active_agents_and_map(self):
        active_agents = []
        active_layerid2listid = {}
        for agent in self.city.agents:
            if agent.layer_id in self.parked_controlled_layers:
                continue
            active_layerid2listid[agent.layer_id] = len(active_agents)
            active_agents.append(agent)
        return active_agents, active_layerid2listid

    def _park_finished_controlled_agents(self):
        for agent in self.controlled_agents:
            if agent.reach_goal and agent.layer_id not in self.parked_controlled_layers:
                self.city.city_grid[agent.layer_id] *= 0
                self.parked_controlled_layers.add(agent.layer_id)

    @staticmethod
    def _path_to_set(path):
        return {tuple(int(v) for v in point.tolist()) for point in path}

    def _resolve_controlled_agents(self):
        selected = []
        for agent_name in self.controlled_agent_names:
            agent_type, agent_id = agent_name.split("_")
            match = None
            for agent in self.city.agents:
                if agent.type == agent_type and agent.id == int(agent_id):
                    match = agent
                    break
            if match is None:
                raise ValueError(f"Controlled agent {agent_name} not found in current city.")
            selected.append(match)
        return selected

    def _macro_action_index(self, action):
        if isinstance(action, (int, np.integer)):
            return int(action)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action)
        for idx, macro_action in self.action_mapping.items():
            macro_action = np.asarray(macro_action, dtype=np.float32)
            if action.shape == macro_action.shape and np.any((action > 0) & (macro_action > 0)):
                return int(idx)
        raise ValueError(f"Could not map action {action} to a discrete macro action.")

    def _flatten_obs(self, grounding, normed_path_length):
        if self.grounding_mode == "thesis_minimal":
            grounding = self._build_thesis_minimal_obs(grounding)
        if self.cat_length:
            return np.concatenate([grounding, [normed_path_length]], axis=0, dtype=np.float32)
        return grounding

    def _build_thesis_minimal_obs(self, grounding):
        entity_slots = int(self.rl_agent_cfg["fov_entities"]["Entity"])

        def _binary_idx(first_idx, second_idx):
            return first_idx * entity_slots + second_idx

        ego_in_inter = float(grounding.get("IsInInter_0", 0.0))
        other_in_inter = max(
            max(
                float(grounding.get(f"IsCar_{idx}", 0.0)),
                float(grounding.get(f"IsPedestrian_{idx}", 0.0)),
            )
            * float(grounding.get(f"IsInInter_{idx}", 0.0))
            for idx in range(1, entity_slots)
        )
        ego_at_inter = float(grounding.get("IsAtInter_0", 0.0))
        close_ahead = max(
            float(grounding.get(f"IsCloseAhead_{_binary_idx(0, idx)}", 0.0))
            for idx in range(1, entity_slots)
        )
        ahead = max(
            float(grounding.get(f"IsAhead_{_binary_idx(0, idx)}", 0.0))
            for idx in range(1, entity_slots)
        )
        higher_pri = max(
            max(
                float(grounding.get(f"IsCar_{idx}", 0.0)),
                float(grounding.get(f"IsPedestrian_{idx}", 0.0)),
            )
            * max(
                float(grounding.get(f"IsAtInter_{idx}", 0.0)),
                float(grounding.get(f"IsInInter_{idx}", 0.0)),
            )
            * float(grounding.get(f"HigherPri_{_binary_idx(idx, 0)}", 0.0))
            for idx in range(1, entity_slots)
        )

        return np.asarray(
            [ego_in_inter, other_in_inter, ego_at_inter, close_ahead, ahead, higher_pri],
            dtype=np.float32,
        )

    def _collect_agent_view(self, agent):
        active_agents, active_layerid2listid = self._active_agents_and_map()
        results = self.city.local_planner.plan(
            self.city.city_grid.clone(),
            self.city.intersection_matrix,
            active_agents,
            active_layerid2listid,
            use_multiprocessing=self.city.use_multi,
            rl_agent=agent.layer_id,
        )
        agent_key = f"{agent.type}_{agent.layer_id}"
        grounding = results[f"{agent_key}_grounding"]
        grounding_dic = results.get(f"{agent_key}_grounding_dic")
        expert_action = results.get(f"{agent_key}_action")
        expert_idx = None
        if expert_action is not None:
            expert_idx = self._macro_action_index(expert_action)
        obs_grounding = grounding_dic if (self.grounding_mode == "thesis_minimal" and grounding_dic is not None) else grounding
        return {
            "obs": self._flatten_obs(obs_grounding, self.normed_path_lengths[agent.layer_id]),
            "grounding": grounding,
            "grounding_dic": grounding_dic,
            "expert_action": expert_idx,
        }

    def collect_views(self):
        self.city.local_planner.reset()
        return {
            agent.layer_id: self._collect_agent_view(agent)
            for agent in self.controlled_agents
            if agent.layer_id not in self.parked_controlled_layers
        }

    def passes_quality_filter(self):
        ped_agents = [agent for agent in self.city.agents if agent.type == "Pedestrian"]
        require_ped_route_intersection = bool(self.rl_agent_cfg.get("require_ped_route_intersection", False))
        require_car_route_intersection = bool(self.rl_agent_cfg.get("require_car_route_intersection", False))
        if require_ped_route_intersection and ped_agents:
            ped_paths = [self._path_to_set(agent.global_traj) for agent in ped_agents]
            for agent in self.controlled_agents:
                car_path = self._path_to_set(agent.global_traj)
                if not any(car_path.intersection(ped_path) for ped_path in ped_paths):
                    return False
        if require_car_route_intersection and len(self.controlled_agents) >= 2:
            shared = self._path_to_set(self.controlled_agents[0].global_traj).intersection(
                self._path_to_set(self.controlled_agents[1].global_traj)
            )
            if len(shared) == 0:
                return False
        return True

    def _reward_for_action(self, agent, grounding, discrete_action):
        one_hot_action = torch.tensor(self.action_mapping[int(discrete_action)], dtype=torch.float32)
        fail, sat_reward = self.city.local_planner.eval_state_action(grounding, one_hot_action)
        if fail:
            return float(sat_reward), bool(fail)
        moving_cost = self.action_cost[int(discrete_action)]
        return float((moving_cost + sat_reward) / self.path_lengths[agent.layer_id]), bool(fail)

    def evaluate_joint_candidate(self, current_views, actions):
        probe_runner = copy.deepcopy(self)
        probe_views = copy.deepcopy(current_views)
        _, rewards, done, info = probe_runner.step(probe_views, {int(k): int(v) for k, v in actions.items()})
        return {
            "done": bool(done),
            "any_fail": bool(info.get("any_fail", False)),
            "overtime": bool(info.get("overtime", False)),
            "joint_success": bool(info.get("is_success", False)),
            "mean_reward": float(np.mean(list(rewards.values()))) if rewards else 0.0,
            "agent_success": dict(info.get("agent_success", {})),
        }

    def _move_controlled_agents(self, actions):
        for agent in self.controlled_agents:
            if agent.reach_goal:
                continue
            discrete_action = int(actions[agent.layer_id])
            one_hot_action = torch.tensor(self.action_mapping[discrete_action], dtype=torch.float32)
            local_action, next_layer = agent.get_next_action(self.city.city_grid, one_hot_action)
            if not agent.reach_goal:
                next_layer = agent.move(local_action, next_layer)
            self.city.city_grid[agent.layer_id] = next_layer
        self._park_finished_controlled_agents()

    def _move_uncontrolled_agents(self):
        active_agents, active_layerid2listid = self._active_agents_and_map()
        agent_action_dist = self.city.local_planner.plan(
            self.city.city_grid.clone(),
            self.city.intersection_matrix,
            active_agents,
            active_layerid2listid,
            use_multiprocessing=self.city.use_multi,
            rl_agent=None,
        )
        new_layers = {}
        for agent in active_agents:
            if agent.layer_id in self.controlled_layer_ids:
                continue
            agent_key = f"{agent.type}_{agent.layer_id}"
            local_action_dist = agent_action_dist[agent_key]
            local_action, next_layer = agent.get_next_action(self.city.city_grid, local_action_dist)
            if not agent.reach_goal:
                next_layer = agent.move(local_action, next_layer)
            new_layers[agent.layer_id] = next_layer
        for layer_id, next_layer in new_layers.items():
            self.city.city_grid[layer_id] = next_layer

    def step(self, current_views, actions):
        self.t += 1
        prev_world = self.city.city_grid.clone()
        rewards = {}
        fails = {}
        expert_actions = {}
        rule_based_fail_count = 0
        for agent in self.controlled_agents:
            layer_id = agent.layer_id
            if layer_id in self.parked_controlled_layers:
                rewards[layer_id] = 0.0
                fails[layer_id] = False
                expert_actions[layer_id] = None
                continue
            expert_actions[layer_id] = current_views[layer_id]["expert_action"]
            reward, fail = self._reward_for_action(agent, current_views[layer_id]["grounding"], actions[layer_id])
            ego_entity = self._entity_name_from_layer(layer_id)
            active_agents, _ = self._active_agents_and_map()
            stop_required = int(
                global_stop_required_conflict(
                    prev_world,
                    self.city.intersection_matrix,
                    active_agents,
                    ego_entity,
                )
            )
            rule_based_fail = int(stop_required and int(actions[layer_id]) != self.stop_action_id)
            if rule_based_fail > 0:
                reward += -10 * rule_based_fail
                fail = True
                rule_based_fail_count += rule_based_fail
            rewards[layer_id] = reward
            fails[layer_id] = fail

        self._move_controlled_agents(actions)
        self._move_uncontrolled_agents()

        active_agents, _ = self._active_agents_and_map()
        hard_breakdown = global_hard_failure_breakdown(
            prev_world,
            self.city.city_grid,
            self.city.intersection_matrix,
            active_agents,
        )
        deadzone_fail_count = int(hard_breakdown["deadzone"])
        simultaneous_entry_fail_count = int(hard_breakdown["simultaneous_entry"])
        event_hard_fail_count = deadzone_fail_count + simultaneous_entry_fail_count
        if event_hard_fail_count > 0:
            for layer_id in rewards:
                rewards[layer_id] += -10 * event_hard_fail_count

        overtime = self.t >= self.horizon
        all_success = all(agent.reach_goal for agent in self.controlled_agents)
        any_fail = any(fails.values()) or (event_hard_fail_count > 0)
        done = all_success or any_fail or overtime
        if overtime:
            for layer_id in rewards:
                rewards[layer_id] += self.overtime_cost

        next_views = self.collect_views() if not done else {}
        info = {
            "is_success": all_success and not any_fail and not overtime,
            "overtime": overtime,
            "any_fail": any_fail,
            "rule_based_fail_count": int(rule_based_fail_count),
            "deadzone_fail_count": int(deadzone_fail_count),
            "simultaneous_entry_fail_count": int(simultaneous_entry_fail_count),
            "agent_fail": {self.layer_id_to_name[k]: bool(v) for k, v in fails.items()},
            "agent_success": {
                self.layer_id_to_name[agent.layer_id]: bool(agent.reach_goal) for agent in self.controlled_agents
            },
            "expert_actions": {self.layer_id_to_name[k]: v for k, v in expert_actions.items()},
            "World": self.city.city_grid.clone(),
        }
        return next_views, rewards, done, info


def load_shared_model(config, checkpoint_path):
    rl_config = config["stable_baselines"]
    if "features_extractor_module" in rl_config["policy_kwargs"]:
        features_extractor_class = dynamic_import(
            rl_config["policy_kwargs"]["features_extractor_module"],
            rl_config["policy_kwargs"]["features_extractor_class"],
        )
        policy_kwargs = {
            "features_extractor_class": features_extractor_class,
            "features_extractor_kwargs": rl_config["policy_kwargs"]["features_extractor_kwargs"],
        }
    else:
        policy_kwargs = rl_config["policy_kwargs"]
    algorithm_class = dynamic_import("logicity.rl_agent.alg", rl_config["algorithm"])
    policy_kwargs_use = copy.deepcopy(policy_kwargs)
    return algorithm_class.load(checkpoint_path, **rl_config["hyperparameters"], policy_kwargs=policy_kwargs_use)


def sample_filtered_city(simulation_config, logger):
    controlled_agent_names = _resolve_controlled_agent_names(simulation_config["rl_agent"])
    attempts = int(simulation_config["rl_agent"].get("sample_attempts", 1))
    last_city = None
    last_cache = None
    for attempt in range(1, attempts + 1):
        city, cached_observation = build_city(simulation_config)
        runner = SharedPolicyTwoAgentRunner(city, simulation_config["rl_agent"], controlled_agent_names)
        if runner.passes_quality_filter():
            logger.info("Accepted two-agent scenario after %s attempt(s).", attempt)
            return city, cached_observation, runner
        last_city = city
        last_cache = cached_observation
    logger.info("Failed to satisfy two-agent quality filter after %s attempt(s); using last sampled scenario.", attempts)
    runner = SharedPolicyTwoAgentRunner(last_city, simulation_config["rl_agent"], controlled_agent_names)
    return last_city, last_cache, runner


def build_runner_from_episode(simulation_config, episode_cache):
    controlled_agent_names = _resolve_controlled_agent_names(simulation_config["rl_agent"])
    city, cached_observation = build_city(simulation_config, episode_cache=episode_cache)
    runner = SharedPolicyTwoAgentRunner(city, simulation_config["rl_agent"], controlled_agent_names)
    return city, cached_observation, runner


def _resolve_controlled_agent_names(rl_agent_cfg):
    controlled_agent_names = rl_agent_cfg.get("agent_names")
    if controlled_agent_names:
        return list(controlled_agent_names)
    single_agent_name = rl_agent_cfg.get("agent_name")
    if single_agent_name == "Car_1":
        return ["Car_1", "Car_2"]
    raise ValueError("Shared runner requires simulation.rl_agent.agent_names or an agent_name compatible with the default two-car setup.")


def _episode_dsr(correct, total):
    total_sum = int(sum(total.values()))
    if total_sum == 0:
        return None
    return float(all(int(correct[name]) == int(total[name]) for name in total.keys()))


def _model_base_action_probs(model, obs):
    obs_tensor = obs_as_tensor(obs, model.device)
    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)
        base_dist = getattr(distribution, "distribution", distribution)
        base_probs = base_dist.probs.detach().cpu().numpy()
    return np.asarray(base_probs, dtype=np.float32).reshape(-1)


def _joint_facts_for_active_agents(runner, views, active_agents):
    if len(active_agents) != 2:
        return {"both_at_inter": False, "same_priority": False}

    both_at_inter = True
    priorities = []
    for agent in active_agents:
        agent_view = views[agent.layer_id]
        grounding_dic = agent_view.get("grounding_dic") or {}
        both_at_inter = both_at_inter and bool(float(grounding_dic.get("IsAtInter_0", 0.0)) > 0.5)
        priorities.append(int(runner.layer_id_to_priority[agent.layer_id]))

    return {
        "both_at_inter": bool(both_at_inter),
        "same_priority": bool(priorities[0] == priorities[1]),
    }


def _build_joint_selector(config, runner):
    shield_cfg = config.get("shield") or {}
    if not bool(shield_cfg.get("joint_action_selection", False)):
        return None
    if len(runner.controlled_agents) != 2:
        raise ValueError("Joint two-car shield requires exactly two controlled agents.")
    action_space = shield_cfg.get("action_space")
    return JointProbabilisticLogicShieldTwoCar(
        num_actions=int(runner.num_actions),
        stop_action=int(runner.stop_action_id),
        action_space=action_space,
        graded_safety_weights=shield_cfg.get("graded_safety_weights"),
    )


def run_single_episode(model, runner, cached_observation, max_steps, episode_label, save_worlds, use_expert_policy=False, joint_selector=None):
    views = runner.collect_views()
    cached_observation["Time_Obs"][0] = {"World": runner.city.city_grid.clone()}
    total_reward = 0.0
    step = 0
    done = False
    local_policy_stop = {name: 0 for name in runner.controlled_agent_names}
    local_expert_stop = {name: 0 for name in runner.controlled_agent_names}
    local_matched_stop = {name: 0 for name in runner.controlled_agent_names}
    local_policy_hist = {name: {action_idx: 0 for action_idx in range(runner.num_actions)} for name in runner.controlled_agent_names}
    local_expert_hist = {name: {action_idx: 0 for action_idx in range(runner.num_actions)} for name in runner.controlled_agent_names}
    termination_reason = "unknown"

    logger.info("Starting shared episode %s with max_steps=%s", episode_label, max_steps)

    while (not done) and (step < max_steps):
        step += 1
        actions = {}
        active_agents = [agent for agent in runner.controlled_agents if agent.layer_id not in runner.parked_controlled_layers]
        if use_expert_policy:
            for agent in active_agents:
                agent_name = runner.layer_id_to_name[agent.layer_id]
                agent_view = views[agent.layer_id]
                expert_action = agent_view["expert_action"]
                if expert_action is not None:
                    local_expert_hist[agent_name][int(expert_action)] = local_expert_hist[agent_name].get(int(expert_action), 0) + 1
                    if int(expert_action) == runner.stop_action_id:
                        local_expert_stop[agent_name] += 1
                action_int = int(runner.stop_action_id if expert_action is None else expert_action)
                local_policy_hist[agent_name][action_int] = local_policy_hist[agent_name].get(action_int, 0) + 1
                if action_int == runner.stop_action_id:
                    local_policy_stop[agent_name] += 1
                if expert_action is not None and int(expert_action) == runner.stop_action_id and action_int == runner.stop_action_id:
                    local_matched_stop[agent_name] += 1
                actions[agent.layer_id] = action_int
        elif joint_selector is not None and len(active_agents) == 2:
            base_probs = []
            expert_actions = {}
            for agent in active_agents:
                agent_name = runner.layer_id_to_name[agent.layer_id]
                agent_view = views[agent.layer_id]
                expert_action = agent_view["expert_action"]
                expert_actions[agent.layer_id] = expert_action
                if expert_action is not None:
                    local_expert_hist[agent_name][int(expert_action)] = local_expert_hist[agent_name].get(int(expert_action), 0) + 1
                    if int(expert_action) == runner.stop_action_id:
                        local_expert_stop[agent_name] += 1
                base_probs.append(_model_base_action_probs(model, agent_view["obs"]))

            candidate_infos = {}
            for a0 in range(runner.num_actions):
                for a1 in range(runner.num_actions):
                    candidate_actions = {
                        active_agents[0].layer_id: int(a0),
                        active_agents[1].layer_id: int(a1),
                    }
                    candidate_infos[(int(a0), int(a1))] = runner.evaluate_joint_candidate(views, candidate_actions)

            joint_facts = _joint_facts_for_active_agents(runner, views, active_agents)
            tiebreak_order = tuple(int(agent.layer_id) for agent in active_agents)

            chosen_pair, _ = joint_selector.select_joint_action(
                base_probs_a=base_probs[0],
                base_probs_b=base_probs[1],
                candidate_infos=candidate_infos,
                joint_facts=joint_facts,
                tiebreak_order=tiebreak_order,
                deterministic=True,
            )
            for idx, agent in enumerate(active_agents):
                agent_name = runner.layer_id_to_name[agent.layer_id]
                action_int = int(chosen_pair[idx])
                expert_action = expert_actions[agent.layer_id]
                local_policy_hist[agent_name][action_int] = local_policy_hist[agent_name].get(action_int, 0) + 1
                if action_int == runner.stop_action_id:
                    local_policy_stop[agent_name] += 1
                if expert_action is not None and int(expert_action) == runner.stop_action_id and action_int == runner.stop_action_id:
                    local_matched_stop[agent_name] += 1
                actions[agent.layer_id] = action_int
        else:
            for agent in active_agents:
                agent_name = runner.layer_id_to_name[agent.layer_id]
                agent_view = views[agent.layer_id]
                expert_action = agent_view["expert_action"]
                if expert_action is not None:
                    local_expert_hist[agent_name][int(expert_action)] = local_expert_hist[agent_name].get(int(expert_action), 0) + 1
                    if int(expert_action) == runner.stop_action_id:
                        local_expert_stop[agent_name] += 1
                obs = agent_view["obs"]
                action, _ = model.predict(obs, deterministic=True)
                action_int = int(action)
                local_policy_hist[agent_name][action_int] = local_policy_hist[agent_name].get(action_int, 0) + 1
                if action_int == runner.stop_action_id:
                    local_policy_stop[agent_name] += 1
                if expert_action is not None and int(expert_action) == runner.stop_action_id and action_int == runner.stop_action_id:
                    local_matched_stop[agent_name] += 1
                actions[agent.layer_id] = action_int
        views, rewards, done, info = runner.step(views, actions)
        total_reward += float(np.mean(list(rewards.values())))
        if save_worlds:
            cached_observation["Time_Obs"][step] = {"World": runner.city.city_grid.clone()}
        if step == 1 or step % 25 == 0 or done:
            logger.info(
                "Shared episode %s progress: step=%s/%s done=%s success=%s fail=%s overtime=%s",
                episode_label,
                step,
                max_steps,
                done,
                info.get("is_success", False),
                info.get("any_fail", False),
                info.get("overtime", False),
            )
        if info["any_fail"]:
            termination_reason = "fail"
        elif info["overtime"]:
            termination_reason = "overtime"
        elif info["is_success"]:
            termination_reason = "success"

    if step >= max_steps and not done:
        termination_reason = "truncated"
    elif termination_reason == "unknown" and not info["is_success"]:
        termination_reason = "failed"

    dsr_per_agent = {}
    for name in runner.controlled_agent_names:
        total = local_expert_stop[name]
        dsr_per_agent[name] = None if total == 0 else float(local_matched_stop[name] / total)
    joint_dsr_episode = _episode_dsr(local_matched_stop, local_expert_stop)

    result = {
        "episode": episode_label,
        "steps": int(step),
        "mean_reward": float(total_reward),
        "joint_success": int(info["is_success"]),
        "termination_reason": termination_reason,
        "overtime": int(info["overtime"]),
        "any_fail": int(info["any_fail"]),
        "agent_success": dict(info["agent_success"]),
        "policy_stop_count": local_policy_stop,
        "expert_stop_count": local_expert_stop,
        "matched_stop_count": local_matched_stop,
        "policy_action_hist": local_policy_hist,
        "expert_action_hist": local_expert_hist,
        "dsr_per_agent": dsr_per_agent,
        "joint_dsr_episode": joint_dsr_episode,
    }
    return result


def _write_episode_csv(csv_path, episode_rows):
    fieldnames = []
    for row in episode_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in episode_rows:
            writer.writerow({k: _jsonable(v) for k, v in row.items()})


def _write_summary_csv(csv_path, summary_row):
    fieldnames = list(summary_row.keys())
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({k: _jsonable(v) for k, v in summary_row.items()})


def main(args, logger):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = load_config(args.config)
    simulation_config = config["simulation"]
    rl_agent_cfg = simulation_config["rl_agent"]
    controlled_agent_names = _resolve_controlled_agent_names(rl_agent_cfg)
    if len(controlled_agent_names) != 2:
        raise ValueError("This tool expects exactly two controlled agents.")

    algorithm_name = str(config["stable_baselines"]["algorithm"])
    use_expert_policy = algorithm_name == "ExpertCollector"
    checkpoint_path = args.checkpoint_path or config["stable_baselines"].get("checkpoint_path", "")
    if (not use_expert_policy) and (not os.path.isfile(checkpoint_path)):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    episode_data_path = args.episode_data or config["stable_baselines"].get("episode_data")
    episode_data = None
    if episode_data_path:
        if not os.path.isfile(episode_data_path):
            raise FileNotFoundError(f"Episode data not found: {episode_data_path}")
        with open(episode_data_path, "rb") as f:
            episode_data = pkl.load(f)
        logger.info("Loaded %s cached episodes from %s", len(episode_data), episode_data_path)

    world_dir = os.path.join(args.log_dir, f"{args.exp}_worlds")
    if args.save_worlds:
        os.makedirs(world_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    results_dir = os.path.join("results", args.exp)
    episode_metrics_path = os.path.join(results_dir, "episode_metrics.csv")
    summary_metrics_path = os.path.join(results_dir, "summary_metrics.csv")

    episode_rows = []
    eval_start_time = time.time()
    mean_rewards = []
    joint_success_values = []
    tsr_per_agent_acc = {name: [] for name in controlled_agent_names}
    dsr_per_agent_correct = {name: 0 for name in controlled_agent_names}
    dsr_per_agent_total = {name: 0 for name in controlled_agent_names}
    joint_dsr_success = 0
    joint_dsr_eligible = 0
    fail_reasons = Counter()
    fail_episodes = 0
    timeout_episodes = 0
    truncated_episodes = 0
    oracle_steps = []
    ppo_steps = []
    agg_policy_stop = {name: 0 for name in controlled_agent_names}
    agg_expert_stop = {name: 0 for name in controlled_agent_names}
    agg_matched_stop = {name: 0 for name in controlled_agent_names}
    action_count = int(rl_agent_cfg["action_space"])
    agg_policy_hist = {name: {action_idx: 0 for action_idx in range(action_count)} for name in controlled_agent_names}
    agg_expert_hist = {name: {action_idx: 0 for action_idx in range(action_count)} for name in controlled_agent_names}

    if episode_data is not None:
        episode_items = list(episode_data.items())
    else:
        episode_items = [(idx, None) for idx in range(args.episodes)]

    if use_expert_policy:
        model = None
        logger.info("Running shared expert policy rollout without checkpoint.")
    else:
        model = load_shared_model(config, checkpoint_path)
        logger.info("Loaded shared PPO policy from %s", checkpoint_path)

    joint_selector = None

    for episode_idx, episode_cache in episode_items:
        logger.info("Preparing shared episode %s/%s", len(episode_rows) + 1, len(episode_items))
        if episode_cache is None:
            city, cached_observation, runner = sample_filtered_city(simulation_config, logger)
            episode_label = int(episode_idx)
            oracle_step = None
            max_steps = int(rl_agent_cfg["max_horizon"])
        else:
            city, cached_observation, runner = build_runner_from_episode(simulation_config, episode_cache)
            episode_label = episode_idx
            label_info = episode_cache.get("label_info", {})
            oracle_step = label_info.get("joint_oracle_step", label_info.get("oracle_step"))
            if oracle_step is not None:
                oracle_step = int(oracle_step)
                oracle_steps.append(oracle_step)
                max_steps = max(int(np.ceil(oracle_step * 2.5)), 1)
            else:
                max_steps = int(rl_agent_cfg["max_horizon"])
        shield_cfg = config.get("shield")
        if joint_selector is None:
            joint_selector = _build_joint_selector(config, runner)
            if joint_selector is not None:
                logger.info("Enabled joint two-car action shield.")
        if (model is not None) and hasattr(model, "configure_shield") and getattr(model, "_shield", None) is None:
            model.configure_shield(runner.city.pred_grounding_index, shield_cfg)
        if (model is not None) and hasattr(model, "reset_shield_metrics"):
            model.reset_shield_metrics()
        if joint_selector is not None:
            joint_selector.reset_metrics()

        result = run_single_episode(
            model=model,
            runner=runner,
            cached_observation=cached_observation,
            max_steps=max_steps,
            episode_label=episode_label,
            save_worlds=args.save_worlds,
            use_expert_policy=use_expert_policy,
            joint_selector=joint_selector,
        )

        mean_rewards.append(result["mean_reward"])
        joint_success_values.append(result["joint_success"])
        ppo_steps.append(result["steps"])
        for name in controlled_agent_names:
            tsr_per_agent_acc[name].append(int(result["agent_success"].get(name, False)))
            agg_policy_stop[name] += int(result["policy_stop_count"][name])
            agg_expert_stop[name] += int(result["expert_stop_count"][name])
            agg_matched_stop[name] += int(result["matched_stop_count"][name])
            dsr_per_agent_total[name] += int(result["expert_stop_count"][name])
            dsr_per_agent_correct[name] += int(result["matched_stop_count"][name])
            for action_idx in range(action_count):
                agg_policy_hist[name][action_idx] += int(result["policy_action_hist"][name].get(action_idx, 0))
                agg_expert_hist[name][action_idx] += int(result["expert_action_hist"][name].get(action_idx, 0))

        if result["joint_dsr_episode"] is not None:
            joint_dsr_eligible += 1
            joint_dsr_success += int(result["joint_dsr_episode"])

        reason = result["termination_reason"]
        fail_reasons[reason] += 1
        if reason == "fail":
            fail_episodes += 1
        if reason == "overtime":
            timeout_episodes += 1
        if reason == "truncated":
            truncated_episodes += 1

        logger.info(
            "Episode %s | joint_success=%s | joint_dsr=%s | steps=%s | reward=%s | reason=%s",
            episode_label,
            result["joint_success"],
            result["joint_dsr_episode"],
            result["steps"],
            result["mean_reward"],
            reason,
        )
        logger.info("Per-agent success: %s", result["agent_success"])
        logger.info("Per-agent DSR: %s", result["dsr_per_agent"])

        episode_row = {
            "episode": episode_label,
            "joint_success": result["joint_success"],
            "joint_dsr": result["joint_dsr_episode"],
            "mean_reward": result["mean_reward"],
            "ppo_steps": result["steps"],
            "oracle_step": oracle_step,
            "termination_reason": reason,
            "fail": int(reason == "fail"),
            "timeout": int(reason == "overtime"),
            "truncated": int(reason == "truncated"),
        }
        for name in controlled_agent_names:
            safe_name = name.lower()
            episode_row[f"{safe_name}_success"] = int(result["agent_success"].get(name, False))
            episode_row[f"{safe_name}_dsr"] = result["dsr_per_agent"][name]
            episode_row[f"{safe_name}_policy_stop_count"] = int(result["policy_stop_count"][name])
            episode_row[f"{safe_name}_expert_stop_count"] = int(result["expert_stop_count"][name])
            episode_row[f"{safe_name}_matched_stop_count"] = int(result["matched_stop_count"][name])
            for action_idx in range(action_count):
                episode_row[f"{safe_name}_policy_action_hist_{action_idx}"] = int(result["policy_action_hist"][name].get(action_idx, 0))
                episode_row[f"{safe_name}_expert_action_hist_{action_idx}"] = int(result["expert_action_hist"][name].get(action_idx, 0))
        episode_row["eval_wall_clock_seconds"] = 0.0
        episode_rows.append(episode_row)

        if args.save_worlds:
            output_path = os.path.join(world_dir, f"{args.exp}_{episode_label}.pkl")
            with open(output_path, "wb") as f:
                pkl.dump(cached_observation, f)

    summary_row = {
        "joint_tsr": float(np.mean(joint_success_values)) if joint_success_values else 0.0,
        "joint_dsr": float(joint_dsr_success / joint_dsr_eligible) if joint_dsr_eligible > 0 else None,
        "joint_dsr_eligible_episodes": int(joint_dsr_eligible),
        "mean_reward": float(np.mean(mean_rewards)) if mean_rewards else 0.0,
        "fail": int(fail_episodes),
        "timeout": int(timeout_episodes),
        "truncated": int(truncated_episodes),
        "termination_reason_counts": json.dumps(dict(fail_reasons)),
        "shield_intervention_count": 0,
        "shield_intervention_rate": 0.0,
        "eval_wall_clock_seconds": float(time.time() - eval_start_time),
    }
    if (model is not None) and hasattr(model, "get_shield_metrics"):
        summary_row.update(model.get_shield_metrics())
    if joint_selector is not None:
        summary_row.update(joint_selector.get_metrics())
    for name in controlled_agent_names:
        safe_name = name.lower()
        summary_row[f"{safe_name}_tsr"] = float(np.mean(tsr_per_agent_acc[name])) if tsr_per_agent_acc[name] else 0.0
        summary_row[f"{safe_name}_dsr"] = (
            float(dsr_per_agent_correct[name] / dsr_per_agent_total[name])
            if dsr_per_agent_total[name] > 0
            else None
        )
        summary_row[f"{safe_name}_policy_stop_count"] = int(agg_policy_stop[name])
        summary_row[f"{safe_name}_expert_stop_count"] = int(agg_expert_stop[name])
        summary_row[f"{safe_name}_matched_stop_count"] = int(agg_matched_stop[name])
        for action_idx in range(action_count):
            summary_row[f"{safe_name}_policy_action_hist_{action_idx}"] = int(agg_policy_hist[name][action_idx])
            summary_row[f"{safe_name}_expert_action_hist_{action_idx}"] = int(agg_expert_hist[name][action_idx])

    _write_episode_csv(episode_metrics_path, episode_rows)
    _write_summary_csv(summary_metrics_path, summary_row)
    logger.info("Saved episode metrics to %s", episode_metrics_path)
    logger.info("Saved summary metrics to %s", summary_metrics_path)
    logger.info("Joint TSR: %s", summary_row["joint_tsr"])
    logger.info("Joint DSR: %s", summary_row["joint_dsr"])
    for name in controlled_agent_names:
        safe_name = name.lower()
        logger.info("%s TSR: %s", name, summary_row[f"{safe_name}_tsr"])
        logger.info("%s DSR: %s", name, summary_row[f"{safe_name}_dsr"])


if __name__ == "__main__":
    args = parse_arguments()
    logger = setup_logger(log_dir=args.log_dir, log_name=args.exp)
    main(args, logger)
