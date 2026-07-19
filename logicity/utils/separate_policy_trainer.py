import copy
import os
import pickle as pkl
from itertools import permutations

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import obs_as_tensor

from logicity.utils.gym_callback import (
    ACTION_ID_TO_NAME,
    _build_per_agent_eval_row,
    _load_task_rule_names,
    _sanitize_rule_name,
    _update_per_agent_episode_stats,
    append_eval_csv_row,
)
from logicity.utils.load import CityLoader
from logicity.utils.multi_agent_wrapper import SharedPolicyMultiAgentVecEnv


class _BootstrapSingleAgentEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, observation_space, action_space, pred_grounding_index):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.pred_grounding_index = pred_grounding_index

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, 0.0, True, False, {}


def _make_shared_env(simulation_config, episode_cache=None):
    city, _ = CityLoader.from_yaml(**simulation_config, episode_cache=episode_cache)
    return SharedPolicyMultiAgentVecEnv(city)


def _load_single_separate_policy_model(
    algorithm_class,
    rl_config,
    hyperparameters,
    policy_kwargs,
    observation_space,
    action_space,
    pred_grounding_index,
    checkpoint_path,
    seed,
):
    bootstrap_env = _BootstrapSingleAgentEnv(
        observation_space,
        action_space,
        pred_grounding_index,
    )
    model = algorithm_class.load(
        checkpoint_path,
        bootstrap_env,
        **copy.deepcopy(hyperparameters),
        policy_kwargs=copy.deepcopy(policy_kwargs),
    )
    model.set_random_seed(seed)
    return model


def _infer_difficulty_from_config_path(config_path):
    normalized = str(config_path).replace("\\", "/")
    for difficulty in ["easy", "medium", "hard"]:
        if f"/{difficulty}/" in normalized:
            return difficulty
    return "unknown"


def _result_file_path(args, filename):
    results_dir = os.path.join(args.results_dir, args.exp)
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, filename)


def _cal_step_metric(decision_step, succ_decision):
    mean_decision_succ = {}
    total_decision = max(sum(decision_step.values()), 1)
    total_succ = sum(succ_decision.values())
    for action_id, num in decision_step.items():
        mean_decision_succ[action_id] = succ_decision[action_id] / max(num, 1)
    average_decision_succ = sum(mean_decision_succ.values()) / max(len(mean_decision_succ), 1)
    return total_succ / total_decision, average_decision_succ, mean_decision_succ


def _append_plpg_stats(plpg_info, stats):
    if plpg_info is None:
        return
    base_actions = np.atleast_1d(plpg_info["base_action"])
    shielded_actions = np.atleast_1d(plpg_info["shielded_action"])
    for base_action in base_actions:
        stats["base_action_hist"][int(base_action)] += 1
    for shielded_action in shielded_actions:
        stats["shielded_action_hist"][int(shielded_action)] += 1
    stats["safe_prob_sums"] += np.asarray(plpg_info["safety_probs"], dtype=np.float64).sum(axis=0)
    stats["base_safe_prob_sum"] += float(np.asarray(plpg_info["base_policy_safe_prob"], dtype=np.float64).sum())
    stats["shielded_safe_prob_sum"] += float(np.asarray(plpg_info["shielded_policy_safe_prob"], dtype=np.float64).sum())
    stats["intervention_count"] += float(np.asarray(plpg_info["intervened"], dtype=np.float64).sum())
    stats["kl_sum"] += float(np.asarray(plpg_info["kl_base_to_shielded"], dtype=np.float64).sum())
    stats["l1_sum"] += float(np.asarray(plpg_info["l1_shift_base_to_shielded"], dtype=np.float64).sum())
    stats["hazard_count"] += float(np.asarray(plpg_info["hazard"], dtype=np.float64).sum())
    stats["forced_stop_count"] += float(np.asarray(plpg_info["forced_stop"], dtype=np.float64).sum())
    base_safe = np.asarray(plpg_info["base_policy_safe_prob"])
    stats["plpg_step_count"] += int(base_safe.shape[0]) if base_safe.ndim > 0 else 1


def _evaluate_separate_policy_models(models, simulation_config, episode_data_path, eval_actions, logger, model_labels=None):
    with open(episode_data_path, "rb") as handle:
        episode_data = pkl.load(handle)

    rewards = []
    joint_successes = []
    joint_failures = []
    joint_timeouts = []
    episode_lengths = []
    mean_agent_successes = []
    completed_rl_agents = []
    completed_rl_agents_before_failure = []
    completed_rl_agents_before_timeout = []
    per_agent_eval_stats = {}
    decision_step = {action_id: 0 for action_id in eval_actions.values()}
    succ_decision = {action_id: 0 for action_id in eval_actions.values()}
    action_hist = {0: 0, 1: 0, 2: 0, 3: 0}
    fail_rl_agent_counts = {}
    fail_rule_counts = {}
    num_rl_agents = len(models)
    model_labels = list(model_labels or [f"agent{idx + 1}" for idx in range(num_rl_agents)])
    per_model_successes = {label: [] for label in model_labels}
    per_model_action_hist = {label: {0: 0, 1: 0, 2: 0, 3: 0} for label in model_labels}
    per_model_interventions = {label: 0.0 for label in model_labels}
    per_model_hazards = {label: 0.0 for label in model_labels}
    per_model_forced_stops = {label: 0.0 for label in model_labels}
    per_model_plpg_steps = {label: 0 for label in model_labels}
    plpg_stats = {
        "base_action_hist": {0: 0, 1: 0, 2: 0, 3: 0},
        "shielded_action_hist": {0: 0, 1: 0, 2: 0, 3: 0},
        "safe_prob_sums": np.zeros(4, dtype=np.float64),
        "base_safe_prob_sum": 0.0,
        "shielded_safe_prob_sum": 0.0,
        "intervention_count": 0.0,
        "kl_sum": 0.0,
        "l1_sum": 0.0,
        "hazard_count": 0.0,
        "forced_stop_count": 0.0,
        "plpg_step_count": 0,
    }

    for episode_id in list(episode_data.keys()):
        episode_cache = episode_data[episode_id]
        eval_env = _make_shared_env(simulation_config, episode_cache=episode_cache)
        obs = eval_env.init()
        done = False
        total_reward = 0.0
        step_count = 0
        max_steps = eval_env.horizon
        final_infos = None
        local_decision_step = {action_id: 0 for action_id in eval_actions.values()}
        local_succ_decision = {action_id: 1 for action_id in eval_actions.values()}

        while (not done) and (step_count < max_steps):
            actions = []
            oracle_action = eval_env.expert_action if hasattr(eval_env, "expert_action") else None
            for agent_idx, model in enumerate(models):
                action, _, plpg_info = model.predict_with_plpg_info(obs[agent_idx], deterministic=True)
                action_int = int(np.atleast_1d(action)[0])
                label = model_labels[agent_idx]
                actions.append(action_int)
                action_hist[action_int] = action_hist.get(action_int, 0) + 1
                per_model_action_hist[label][action_int] = per_model_action_hist[label].get(action_int, 0) + 1
                _append_plpg_stats(plpg_info, plpg_stats)
                if plpg_info is not None:
                    per_model_interventions[label] += float(np.asarray(plpg_info["intervened"], dtype=np.float64).sum())
                    per_model_hazards[label] += float(np.asarray(plpg_info["hazard"], dtype=np.float64).sum())
                    per_model_forced_stops[label] += float(np.asarray(plpg_info["forced_stop"], dtype=np.float64).sum())
                    base_safe = np.asarray(plpg_info["base_policy_safe_prob"])
                    per_model_plpg_steps[label] += int(base_safe.shape[0]) if base_safe.ndim > 0 else 1
            if oracle_action in local_decision_step:
                local_decision_step[oracle_action] = 1
                if actions[0] != oracle_action:
                    local_succ_decision[oracle_action] = 0
            obs, reward, dones, infos = eval_env.step(np.asarray(actions, dtype=np.int64))
            total_reward += float(np.sum(reward))
            step_count += 1
            done = bool(dones[0])
            final_infos = infos

        if final_infos is None:
            raise RuntimeError("Separate-policy evaluation produced no environment infos.")

        joint_info = final_infos[0]
        rewards.append(total_reward / max(num_rl_agents, 1))
        joint_successes.append(float(joint_info.get("joint_is_success", False)))
        joint_failures.append(float(joint_info.get("joint_failure", False)))
        joint_timeouts.append(float(joint_info.get("joint_timeout", False)))
        episode_lengths.append(step_count)
        mean_agent_success = float(np.mean([float(info.get("agent_is_success", False)) for info in final_infos]))
        mean_agent_successes.append(mean_agent_success)
        completed_agents = float(np.sum([float(info.get("agent_is_success", False)) for info in final_infos]))
        completed_rl_agents.append(completed_agents)
        _update_per_agent_episode_stats(
            per_agent_eval_stats,
            final_infos,
            joint_fail_agent_layer_ids=joint_info.get("joint_fail_agent_layer_ids", []),
            joint_timeout=bool(joint_info.get("joint_timeout", False)),
        )
        for agent_idx, info in enumerate(final_infos):
            per_model_successes[model_labels[agent_idx]].append(float(info.get("agent_is_success", False)))
        if joint_info.get("joint_failure", False):
            completed_rl_agents_before_failure.append(completed_agents)
        if joint_info.get("joint_timeout", False):
            completed_rl_agents_before_timeout.append(completed_agents)
        for layer_id in joint_info.get("joint_fail_agent_layer_ids", []):
            key = "fail_rl_agent_{}".format(layer_id)
            fail_rl_agent_counts[key] = fail_rl_agent_counts.get(key, 0) + 1
        for action_id in eval_actions.values():
            if local_decision_step[action_id] == 0:
                local_succ_decision[action_id] = 0
            decision_step[action_id] += local_decision_step[action_id]
            succ_decision[action_id] += local_succ_decision[action_id]
        for info in final_infos:
            for rule_name in info.get("FailRuleNames", [[]])[0]:
                fail_rule_counts[rule_name] = fail_rule_counts.get(rule_name, 0) + 1

    mean_decision_succ, _, decision_succ_by_action = _cal_step_metric(decision_step, succ_decision)
    mean_completed_rl_agents = float(np.mean(completed_rl_agents)) if completed_rl_agents else 0.0
    mean_completed_rl_fraction = mean_completed_rl_agents / max(num_rl_agents, 1)
    episodes_all_k_success = int(np.sum(joint_successes))
    episodes_joint_failure = int(np.sum(joint_failures))
    episodes_joint_timeout = int(np.sum(joint_timeouts))
    plpg_step_count = plpg_stats["plpg_step_count"]
    metrics = {
        "num_eval_episodes": len(rewards),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "joint_tsr": float(np.mean(joint_successes)) if joint_successes else 0.0,
        "joint_failure_rate": float(np.mean(joint_failures)) if joint_failures else 0.0,
        "joint_timeout_rate": float(np.mean(joint_timeouts)) if joint_timeouts else 0.0,
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "mean_agent_tsr": float(np.mean(mean_agent_successes)) if mean_agent_successes else 0.0,
        "mean_completed_rl_agents": mean_completed_rl_agents,
        "mean_completed_rl_fraction": mean_completed_rl_fraction,
        "mean_completed_rl_agents_before_failure": float(np.mean(completed_rl_agents_before_failure)) if completed_rl_agents_before_failure else 0.0,
        "mean_completed_rl_agents_before_timeout": float(np.mean(completed_rl_agents_before_timeout)) if completed_rl_agents_before_timeout else 0.0,
        "episodes_all_k_success": episodes_all_k_success,
        "episodes_joint_failure": episodes_joint_failure,
        "episodes_joint_timeout": episodes_joint_timeout,
        "mean_decision_succ": mean_decision_succ,
        "decision_succ_by_action": decision_succ_by_action,
        "action_hist": action_hist,
        "num_rl_agents": num_rl_agents,
        "fail_rl_agent_counts": fail_rl_agent_counts,
        "fail_rule_counts": fail_rule_counts,
        "mean_base_policy_safe_prob": (plpg_stats["base_safe_prob_sum"] / plpg_step_count) if plpg_step_count > 0 else 0.0,
        "mean_shielded_policy_safe_prob": (plpg_stats["shielded_safe_prob_sum"] / plpg_step_count) if plpg_step_count > 0 else 0.0,
        "mean_safety_gain": ((plpg_stats["shielded_safe_prob_sum"] - plpg_stats["base_safe_prob_sum"]) / plpg_step_count) if plpg_step_count > 0 else 0.0,
        "shield_intervention_rate": (plpg_stats["intervention_count"] / plpg_step_count) if plpg_step_count > 0 else 0.0,
        "mean_policy_kl_base_to_shielded": (plpg_stats["kl_sum"] / plpg_step_count) if plpg_step_count > 0 else 0.0,
        "mean_l1_shift_base_to_shielded": (plpg_stats["l1_sum"] / plpg_step_count) if plpg_step_count > 0 else 0.0,
        "hazard_step_rate": (plpg_stats["hazard_count"] / plpg_step_count) if plpg_step_count > 0 else 0.0,
        "shield_forced_stop_rate": (plpg_stats["forced_stop_count"] / max(plpg_stats["hazard_count"], 1.0)) if plpg_step_count > 0 else 0.0,
        "base_action_hist": plpg_stats["base_action_hist"],
        "shielded_action_hist": plpg_stats["shielded_action_hist"],
        "mean_safe_prob_action": {
            action_id: (plpg_stats["safe_prob_sums"][action_id] / plpg_step_count) if plpg_step_count > 0 else 0.0
            for action_id in ACTION_ID_TO_NAME.keys()
        },
        "per_model_success_rates": {
            label: float(np.mean(values)) if values else 0.0
            for label, values in per_model_successes.items()
        },
        "per_model_action_hist": per_model_action_hist,
        "per_model_intervention_rate": {
            label: (per_model_interventions[label] / max(per_model_plpg_steps[label], 1))
            for label in model_labels
        },
        "per_model_hazard_rate": {
            label: (per_model_hazards[label] / max(per_model_plpg_steps[label], 1))
            for label in model_labels
        },
        "per_model_forced_stop_rate": {
            label: (per_model_forced_stops[label] / max(per_model_hazards[label], 1.0))
            for label in model_labels
        },
        "per_agent_eval_row": _build_per_agent_eval_row(per_agent_eval_stats),
    }
    logger.info(
        "Separate-policy eval: joint_tsr=%.4f mean_agent_tsr=%.4f failure=%.4f timeout=%.4f",
        metrics["joint_tsr"],
        metrics["mean_agent_tsr"],
        metrics["joint_failure_rate"],
        metrics["joint_timeout_rate"],
    )
    return metrics


def _resolve_assignment_permutations(num_agents, permutation_cfg):
    if permutation_cfg is None:
        return [tuple(range(num_agents))]
    if isinstance(permutation_cfg, str):
        if permutation_cfg.lower() == "all":
            return list(permutations(range(num_agents)))
        if permutation_cfg.lower() == "identity":
            return [tuple(range(num_agents))]
    if isinstance(permutation_cfg, list):
        resolved = []
        for item in permutation_cfg:
            resolved.append(tuple(int(x) for x in item))
        return resolved
    raise ValueError("Unsupported separate_policy_eval.permutations config: {}".format(permutation_cfg))


def evaluate_separate_policy_checkpoints(
    args,
    logger,
    simulation_config,
    rl_config,
    algorithm_class,
    policy_kwargs,
):
    if rl_config["algorithm"] != "PLPGPPO":
        raise ValueError("Separate-policy evaluation is currently implemented only for PLPGPPO.")
    separate_eval_cfg = copy.deepcopy(rl_config.get("separate_policy_eval") or {})
    checkpoint_paths = list(separate_eval_cfg.get("checkpoint_paths") or [])
    if len(checkpoint_paths) == 0:
        raise ValueError("separate_policy_eval.checkpoint_paths must be provided for separate-policy test evaluation.")
    if args.checkpoint_path is not None:
        checkpoint_paths = [part.strip() for part in str(args.checkpoint_path).split(",") if part.strip()]
        logger.info("Overwriting separate-policy checkpoint paths to %s", checkpoint_paths)
    hyperparameters = copy.deepcopy(rl_config["hyperparameters"])
    eval_env = _make_shared_env(simulation_config)
    observation_space = eval_env.observation_space
    action_space = eval_env.action_space
    pred_grounding_index = eval_env.pred_grounding_index
    os.makedirs(args.log_dir, exist_ok=True)
    failure_rule_names = _load_task_rule_names(simulation_config["rule_yaml_file"])

    base_models = []
    base_labels = []
    for model_idx, checkpoint_path in enumerate(checkpoint_paths):
        base_models.append(
            _load_single_separate_policy_model(
                algorithm_class,
                rl_config,
                hyperparameters,
                policy_kwargs,
                observation_space,
                action_space,
                pred_grounding_index,
                checkpoint_path,
                args.seed,
            )
        )
        base_labels.append("agent{}".format(model_idx + 1))
    permutation_list = _resolve_assignment_permutations(len(base_models), separate_eval_cfg.get("permutations"))
    csv_path = _result_file_path(args, "{}_test_metrics.csv".format(args.exp))

    for permutation_index, permutation_tuple in enumerate(permutation_list, start=1):
        assigned_models = [base_models[idx] for idx in permutation_tuple]
        assigned_labels = [base_labels[idx] for idx in permutation_tuple]
        metrics = _evaluate_separate_policy_models(
            assigned_models,
            simulation_config,
            rl_config["episode_data"],
            rl_config["eval_actions"],
            logger,
            model_labels=assigned_labels,
        )
        assignment_name = "perm_" + "_".join(str(idx + 1) for idx in permutation_tuple)
        row = {
            "exp_name": args.exp,
            "assignment_name": assignment_name,
            "model_slot_mapping": ",".join(assigned_labels),
            "permutation_index": permutation_index,
            "method": "{}_separate_policy".format(rl_config["algorithm"]),
            "difficulty": _infer_difficulty_from_config_path(args.config),
            "seed": args.seed,
            "split": "test",
            "eval_index": permutation_index,
            "timestep": 0,
            "wall_clock_time_sec": 0.0,
            "best_tsr_so_far": metrics["joint_tsr"],
            "num_eval_episodes": metrics["num_eval_episodes"],
            "tsr": metrics["joint_tsr"],
            "mean_reward": metrics["mean_reward"],
            "failure_rate": metrics["joint_failure_rate"],
            "timeout_rate": metrics["joint_timeout_rate"],
            "mean_episode_length": metrics["mean_episode_length"],
            "num_rl_agents": metrics["num_rl_agents"],
            "mean_agent_tsr": metrics["mean_agent_tsr"],
            "joint_tsr": metrics["joint_tsr"],
            "joint_failure_rate": metrics["joint_failure_rate"],
            "joint_timeout_rate": metrics["joint_timeout_rate"],
            "episodes_all_k_success": metrics["episodes_all_k_success"],
            "episodes_joint_failure": metrics["episodes_joint_failure"],
            "episodes_joint_timeout": metrics["episodes_joint_timeout"],
            "mean_completed_rl_agents": metrics["mean_completed_rl_agents"],
            "mean_completed_rl_fraction": metrics["mean_completed_rl_fraction"],
            "mean_completed_rl_agents_before_failure": metrics["mean_completed_rl_agents_before_failure"],
            "mean_completed_rl_agents_before_timeout": metrics["mean_completed_rl_agents_before_timeout"],
            "mean_decision_succ": metrics["mean_decision_succ"],
            "rollout_failures_since_last_eval": 0,
            "rollout_timeouts_since_last_eval": 0,
            "rollout_successes_since_last_eval": 0,
            "goal_completion_rate": 0.0,
            "mean_agent_goal_completion_rate": 0.0,
            "failure_rate_per_goal": 0.0,
            "failure_rate_per_1k_steps": 0.0,
            "unfinished_goal_rate": 0.0,
            "completed_goals": 0,
            "failed_goals": 0,
            "active_goals_at_budget": 0,
            "deployment_count": 0,
            "failure_reset_count": 0,
            "resolved_goal_attempts": 0,
            "resolved_goal_quota_per_agent": 0,
            "micro_rollout_tsr": 0.0,
            "micro_rollout_failure_rate": 0.0,
            "macro_rollout_tsr": 0.0,
            "macro_rollout_failure_rate": 0.0,
            "mean_steps_per_resolved_goal_attempt": 0.0,
            "mean_base_policy_safe_prob": metrics["mean_base_policy_safe_prob"],
            "mean_shielded_policy_safe_prob": metrics["mean_shielded_policy_safe_prob"],
            "mean_safety_gain": metrics["mean_safety_gain"],
            "shield_intervention_rate": metrics["shield_intervention_rate"],
            "mean_policy_kl_base_to_shielded": metrics["mean_policy_kl_base_to_shielded"],
            "mean_l1_shift_base_to_shielded": metrics["mean_l1_shift_base_to_shielded"],
            "hazard_step_rate": metrics["hazard_step_rate"],
            "shield_forced_stop_rate": metrics["shield_forced_stop_rate"],
        }
        for action_id, action_name in ACTION_ID_TO_NAME.items():
            row[f"action_{action_name}_count"] = metrics["action_hist"].get(action_id, 0)
            row[f"base_action_{action_name}_count"] = metrics["base_action_hist"].get(action_id, 0)
            row[f"shielded_action_{action_name}_count"] = metrics["shielded_action_hist"].get(action_id, 0)
            row[f"mean_safe_prob_action_{action_name}"] = metrics["mean_safe_prob_action"].get(action_id, 0.0)
        for action_id, value in metrics["decision_succ_by_action"].items():
            action_name = ACTION_ID_TO_NAME.get(action_id, str(action_id))
            row[f"decision_succ_action_{action_name}"] = value
        for label, value in metrics["per_model_success_rates"].items():
            row[f"model_{label}_tsr"] = value
            row[f"model_{label}_intervention_rate"] = metrics["per_model_intervention_rate"].get(label, 0.0)
            row[f"model_{label}_hazard_rate"] = metrics["per_model_hazard_rate"].get(label, 0.0)
            row[f"model_{label}_forced_stop_rate"] = metrics["per_model_forced_stop_rate"].get(label, 0.0)
            for action_id, action_name in ACTION_ID_TO_NAME.items():
                row[f"model_{label}_action_{action_name}_count"] = metrics["per_model_action_hist"][label].get(action_id, 0)
        for key, value in metrics["fail_rl_agent_counts"].items():
            row[key] = value
        row.update(metrics.get("per_agent_eval_row", {}))
        for rule_name in failure_rule_names:
            row[f"fail_rule_{_sanitize_rule_name(rule_name)}"] = metrics["fail_rule_counts"].get(rule_name, 0)
        append_eval_csv_row(csv_path, row, failure_rule_names)
        logger.info(
            "Separate-policy test assignment %s complete: mapping=%s joint_tsr=%.4f",
            assignment_name,
            assigned_labels,
            metrics["joint_tsr"],
        )


def _save_separate_policy_models(models, save_dir, prefix):
    os.makedirs(save_dir, exist_ok=True)
    for agent_idx, model in enumerate(models):
        model.save(os.path.join(save_dir, f"{prefix}_agent{agent_idx + 1}"))


def _init_separate_policy_logger(model, exp_name, tensorboard_root, agent_idx):
    if tensorboard_root is not None:
        os.makedirs(tensorboard_root, exist_ok=True)
        log_dir = os.path.join(tensorboard_root, f"{exp_name}_agent{agent_idx + 1}")
        new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    else:
        new_logger = configure(None, ["stdout"])
    model.set_logger(new_logger)


def train_separate_policy_multi_agent(
    args,
    logger,
    simulation_config,
    rl_config,
    eval_checkpoint_config,
    algorithm_class,
    policy_kwargs,
):
    if rl_config["algorithm"] != "PLPGPPO":
        raise ValueError("Separate-policy multi-agent training is currently implemented only for PLPGPPO.")

    rl_agent_cfg = simulation_config.get("rl_agent", {})
    agent_names = list(rl_agent_cfg.get("agent_names") or [])
    if len(agent_names) < 2:
        raise ValueError("Separate-policy multi-agent training requires at least 2 RL agents.")

    total_timesteps = int(rl_config["total_timesteps"])
    hyperparameters = copy.deepcopy(rl_config["hyperparameters"])
    n_steps = int(hyperparameters["n_steps"])
    num_agents = len(agent_names)
    save_root = args.checkpoint_root if args.checkpoint_root is not None else "./checkpoints"
    exp_save_dir = os.path.join(save_root, args.exp)
    os.makedirs(exp_save_dir, exist_ok=True)
    eval_csv_path = _result_file_path(args, f"{args.exp}_eval_metrics.csv")
    failure_rule_names = _load_task_rule_names(simulation_config["rule_yaml_file"])

    train_env = _make_shared_env(simulation_config)
    models = []
    for agent_idx in range(num_agents):
        bootstrap_env = _BootstrapSingleAgentEnv(
            train_env.observation_space,
            train_env.action_space,
            train_env.pred_grounding_index,
        )
        model = algorithm_class(
            rl_config["policy_network"],
            bootstrap_env,
            **copy.deepcopy(hyperparameters),
            policy_kwargs=copy.deepcopy(policy_kwargs),
            seed=args.seed,
        )
        _init_separate_policy_logger(
            model,
            args.exp,
            hyperparameters.get("tensorboard_log"),
            agent_idx,
        )
        models.append(model)

    initial_state = copy.deepcopy(models[0].policy.state_dict())
    for model in models[1:]:
        model.policy.load_state_dict(initial_state)

    current_obs = train_env.reset()
    last_episode_starts = np.ones(num_agents, dtype=bool)
    num_timesteps = 0
    eval_freq = int(eval_checkpoint_config.get("eval_freq", total_timesteps))
    save_freq = int(eval_checkpoint_config.get("save_freq", eval_freq))
    next_eval = eval_freq
    next_save = save_freq
    eval_index = 0
    best_joint_tsr = -np.inf

    for model in models:
        model.num_timesteps = 0
        model._current_progress_remaining = 1.0
        model.rollout_buffer.reset()

    logger.info(
        "Running separate-policy multi-agent PLPG with %d agents, identical initialization, total_timesteps=%d.",
        num_agents,
        total_timesteps,
    )

    while num_timesteps < total_timesteps:
        for model in models:
            model.rollout_buffer.reset()

        rollout_joint_steps = min(n_steps, max((total_timesteps - num_timesteps + num_agents - 1) // num_agents, 1))

        for _ in range(rollout_joint_steps):
            action_list = []
            reward_list = []
            done_list = []
            next_obs_batch = None

            for model in models:
                model.policy.set_training_mode(False)

            with th.no_grad():
                step_cache = []
                for agent_idx, model in enumerate(models):
                    obs_i = np.expand_dims(current_obs[agent_idx], axis=0)
                    obs_tensor = obs_as_tensor(obs_i, model.device)
                    _, _, shielded_dist, values = model._base_and_shielded(obs_tensor)
                    actions = shielded_dist.sample()
                    log_probs = shielded_dist.log_prob(actions)
                    step_cache.append((obs_i, actions, values, log_probs))
                    action_list.append(int(actions.cpu().numpy()[0]))

            next_obs_batch, rewards, dones, infos = train_env.step(np.asarray(action_list, dtype=np.int64))
            joint_done = bool(dones[0])

            for agent_idx, model in enumerate(models):
                obs_i, actions, values, log_probs = step_cache[agent_idx]
                actions_buffer = actions.reshape(-1, 1)
                reward_i = np.asarray([rewards[agent_idx]], dtype=np.float32)
                episode_start_i = np.asarray([last_episode_starts[agent_idx]], dtype=bool)
                model.rollout_buffer.add(
                    obs_i,
                    actions_buffer,
                    reward_i,
                    episode_start_i,
                    values,
                    log_probs,
                )

            current_obs = next_obs_batch
            last_episode_starts = np.asarray(dones, dtype=bool)
            num_timesteps += num_agents
            for model in models:
                model.num_timesteps = num_timesteps
                model._current_progress_remaining = max(0.0, 1.0 - (num_timesteps / float(total_timesteps)))

        with th.no_grad():
            for agent_idx, model in enumerate(models):
                last_obs_i = np.expand_dims(current_obs[agent_idx], axis=0)
                last_values = model.policy.predict_values(obs_as_tensor(last_obs_i, model.device))
                model.rollout_buffer.compute_returns_and_advantage(
                    last_values=last_values,
                    dones=np.asarray([last_episode_starts[agent_idx]], dtype=bool),
                )

        for model in models:
            model.train()

        should_save = num_timesteps >= next_save or num_timesteps >= total_timesteps
        should_eval = num_timesteps >= next_eval or num_timesteps >= total_timesteps

        if should_save:
            _save_separate_policy_models(models, exp_save_dir, f"step_{num_timesteps}")
            next_save += save_freq

        if should_eval:
            eval_index += 1
            metrics = _evaluate_separate_policy_models(
                models,
                eval_checkpoint_config["simulation_config"],
                eval_checkpoint_config["episode_data"],
                eval_checkpoint_config["eval_actions"],
                logger,
            )
            best_tsr_candidate = max(best_joint_tsr, metrics["joint_tsr"])
            row = {
                "exp_name": args.exp,
                "method": "{}_separate_policy".format(rl_config["algorithm"]),
                "difficulty": _infer_difficulty_from_config_path(args.config),
                "seed": args.seed,
                "split": eval_checkpoint_config.get("split", "val"),
                "eval_index": eval_index,
                "timestep": num_timesteps,
                "wall_clock_time_sec": 0.0,
                "best_tsr_so_far": best_tsr_candidate,
                "num_eval_episodes": metrics["num_eval_episodes"],
                "tsr": metrics["joint_tsr"],
                "failure_rate": metrics["joint_failure_rate"],
                "timeout_rate": metrics["joint_timeout_rate"],
                "mean_episode_length": metrics["mean_episode_length"],
                "num_rl_agents": metrics["num_rl_agents"],
                "joint_tsr": metrics["joint_tsr"],
                "joint_failure_rate": metrics["joint_failure_rate"],
                "joint_timeout_rate": metrics["joint_timeout_rate"],
                "mean_agent_tsr": metrics["mean_agent_tsr"],
                "mean_completed_rl_agents": metrics["mean_completed_rl_agents"],
                "mean_completed_rl_fraction": metrics["mean_completed_rl_fraction"],
                "mean_completed_rl_agents_before_failure": metrics["mean_completed_rl_agents_before_failure"],
                "mean_completed_rl_agents_before_timeout": metrics["mean_completed_rl_agents_before_timeout"],
                "episodes_all_k_success": metrics["episodes_all_k_success"],
                "episodes_joint_failure": metrics["episodes_joint_failure"],
                "episodes_joint_timeout": metrics["episodes_joint_timeout"],
                "mean_reward": metrics["mean_reward"],
                "mean_decision_succ": metrics["mean_decision_succ"],
                "rollout_failures_since_last_eval": 0,
                "rollout_timeouts_since_last_eval": 0,
                "rollout_successes_since_last_eval": 0,
                "goal_completion_rate": 0.0,
                "mean_agent_goal_completion_rate": 0.0,
                "failure_rate_per_goal": 0.0,
                "failure_rate_per_1k_steps": 0.0,
                "unfinished_goal_rate": 0.0,
                "completed_goals": 0,
                "failed_goals": 0,
                "active_goals_at_budget": 0,
                "deployment_count": 0,
                "failure_reset_count": 0,
                "resolved_goal_attempts": 0,
                "resolved_goal_quota_per_agent": 0,
                "micro_rollout_tsr": 0.0,
                "micro_rollout_failure_rate": 0.0,
                "macro_rollout_tsr": 0.0,
                "macro_rollout_failure_rate": 0.0,
                "mean_steps_per_resolved_goal_attempt": 0.0,
                "mean_base_policy_safe_prob": metrics["mean_base_policy_safe_prob"],
                "mean_shielded_policy_safe_prob": metrics["mean_shielded_policy_safe_prob"],
                "mean_safety_gain": metrics["mean_safety_gain"],
                "shield_intervention_rate": metrics["shield_intervention_rate"],
                "mean_policy_kl_base_to_shielded": metrics["mean_policy_kl_base_to_shielded"],
                "mean_l1_shift_base_to_shielded": metrics["mean_l1_shift_base_to_shielded"],
                "hazard_step_rate": metrics["hazard_step_rate"],
                "shield_forced_stop_rate": metrics["shield_forced_stop_rate"],
            }
            for action_id, action_name in ACTION_ID_TO_NAME.items():
                row[f"action_{action_name}_count"] = metrics["action_hist"].get(action_id, 0)
                row[f"base_action_{action_name}_count"] = metrics["base_action_hist"].get(action_id, 0)
                row[f"shielded_action_{action_name}_count"] = metrics["shielded_action_hist"].get(action_id, 0)
                row[f"mean_safe_prob_action_{action_name}"] = metrics["mean_safe_prob_action"].get(action_id, 0.0)
            for action_id, value in metrics["decision_succ_by_action"].items():
                action_name = ACTION_ID_TO_NAME.get(action_id, str(action_id))
                row[f"decision_succ_action_{action_name}"] = value
            for key, value in metrics["fail_rl_agent_counts"].items():
                row[key] = value
            for rule_name in failure_rule_names:
                row[f"fail_rule_{_sanitize_rule_name(rule_name)}"] = metrics["fail_rule_counts"].get(rule_name, 0)

            append_eval_csv_row(eval_csv_path, row, failure_rule_names)

            if metrics["joint_tsr"] > best_joint_tsr:
                best_joint_tsr = metrics["joint_tsr"]
                _save_separate_policy_models(models, exp_save_dir, "best")
            next_eval += eval_freq

    _save_separate_policy_models(models, exp_save_dir, "final")
