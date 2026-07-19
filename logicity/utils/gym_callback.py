from stable_baselines3.common.callbacks import CheckpointCallback
from logicity.utils.load import CityLoader
from logicity.utils.gym_wrapper import GymCityWrapper
from logicity.utils.multi_agent_wrapper import SharedPolicyMultiAgentVecEnv
import numpy as np
import torch
import os
import logging
import pickle as pkl
import csv
import yaml
import time
logger = logging.getLogger(__name__)
ACTION_ID_TO_NAME = {0: "slow", 1: "normal", 2: "fast", 3: "stop"}


def _unpack_step_result(step_result):
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return obs, reward, (terminated or truncated), info
    return step_result


def _format_prob_vector(prob_vector):
    arr = np.asarray(prob_vector, dtype=np.float64).reshape(-1)
    return {
        ACTION_ID_TO_NAME.get(i, str(i)): round(float(arr[i]), 4)
        for i in range(arr.shape[0])
    }


def _sanitize_metric_name(name):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name)).strip("_")


def _update_per_agent_episode_stats(per_agent_stats, final_infos, joint_fail_agent_layer_ids=None, joint_timeout=False):
    joint_fail_agent_layer_ids = set(joint_fail_agent_layer_ids or [])
    for agent_idx, info in enumerate(final_infos):
        layer_id = int(info.get("rl_agent_layer_id", agent_idx))
        agent_name = str(info.get("rl_agent_name", "agent_{}".format(agent_idx)))
        stats = per_agent_stats.setdefault(
            agent_name,
            {
                "episodes": 0,
                "success_sum": 0.0,
                "reward_sum": 0.0,
                "active_steps_sum": 0.0,
                "timeout_unresolved_count": 0,
                "failure_involvement_count": 0,
            },
        )
        stats["episodes"] += 1
        success_value = float(info.get("agent_is_success", False))
        stats["success_sum"] += success_value
        episode_info = info.get("episode", {}) or {}
        stats["reward_sum"] += float(episode_info.get("r", info.get("agent_reward", 0.0)))
        stats["active_steps_sum"] += float(episode_info.get("l", 0.0))
        if joint_timeout and not bool(info.get("agent_is_success", False)):
            stats["timeout_unresolved_count"] += 1
        if layer_id in joint_fail_agent_layer_ids:
            stats["failure_involvement_count"] += 1


def _build_per_agent_eval_row(per_agent_stats):
    row = {}
    for agent_name in sorted(per_agent_stats.keys()):
        stats = per_agent_stats[agent_name]
        denom = max(int(stats.get("episodes", 0)), 1)
        prefix = "agent_{}".format(_sanitize_metric_name(agent_name))
        row["{}_tsr".format(prefix)] = float(stats.get("success_sum", 0.0)) / denom
        row["{}_mean_reward".format(prefix)] = float(stats.get("reward_sum", 0.0)) / denom
        row["{}_mean_active_steps".format(prefix)] = float(stats.get("active_steps_sum", 0.0)) / denom
        row["{}_timeout_unresolved_rate".format(prefix)] = float(stats.get("timeout_unresolved_count", 0)) / denom
        row["{}_failure_involvement_rate".format(prefix)] = float(stats.get("failure_involvement_count", 0)) / denom
    return row


def _extract_easy_debug_facts(obs, pred_grounding_index):
    if pred_grounding_index is None:
        return []
    required = ["IsAtInter", "IsInInter", "HigherPri", "CollidingClose"]
    if any(key not in pred_grounding_index for key in required):
        return []

    obs_array = np.asarray(obs, dtype=np.float32)
    if obs_array.ndim == 1:
        obs_array = obs_array.reshape(1, -1)

    is_at_start, is_at_end = pred_grounding_index["IsAtInter"]
    is_in_start, _ = pred_grounding_index["IsInInter"]
    higher_start, _ = pred_grounding_index["HigherPri"]
    colliding_start, colliding_end = pred_grounding_index["CollidingClose"]
    num_entities = is_at_end - is_at_start

    def pair_index(i, j):
        return i * num_entities + j

    logic_slice_end = colliding_end
    snapshots = []
    for agent_obs in obs_array:
        logic_obs = agent_obs[:logic_slice_end]
        facts = {"ego_is_at_inter": int(round(float(logic_obs[is_at_start])))}
        for i in range(1, num_entities):
            facts[f"other_{i}_is_at_inter"] = int(round(float(logic_obs[is_at_start + i])))
            facts[f"other_{i}_is_in_inter"] = int(round(float(logic_obs[is_in_start + i])))
            facts[f"other_{i}_higher_pri"] = int(round(float(logic_obs[higher_start + pair_index(i, 0)])))
            facts[f"other_{i}_colliding_close"] = int(round(float(logic_obs[colliding_start + pair_index(0, i)])))
        snapshots.append(facts)
    return snapshots


class AdaptiveEvalSchedule:
    def __init__(self, config, default_eval_freq):
        config = config or {}
        self.enabled = bool(config)
        self.default_eval_freq = int(default_eval_freq)
        self.target_tsr = float(config.get("target_tsr", 1.0))
        self.consecutive_target_evals = int(config.get("consecutive_target_evals", 2))
        self.early_eval_freq = int(config.get("early_eval_freq", default_eval_freq))
        self.post_target_eval_offsets = [
            int(offset) for offset in config.get("post_target_eval_offsets", [])
        ]
        self.late_eval_freq = int(config.get("late_eval_freq", default_eval_freq))
        self._target_streak = 0
        self._transition_timestep = None
        self._post_target_eval_index = 0

    def should_eval(self, current_timestep, last_eval_timestep):
        if not self.enabled:
            return (current_timestep - last_eval_timestep) >= self.default_eval_freq
        if self._transition_timestep is None:
            return (current_timestep - last_eval_timestep) >= self.early_eval_freq
        if self._post_target_eval_index < len(self.post_target_eval_offsets):
            due_timestep = self._transition_timestep + self.post_target_eval_offsets[self._post_target_eval_index]
            return current_timestep >= due_timestep
        return (current_timestep - last_eval_timestep) >= self.late_eval_freq

    def on_eval_result(self, current_timestep, tsr):
        if not self.enabled or self._transition_timestep is not None:
            return
        if tsr >= self.target_tsr:
            self._target_streak += 1
        else:
            self._target_streak = 0
        if self._target_streak >= self.consecutive_target_evals:
            self._transition_timestep = current_timestep
            self._post_target_eval_index = 0

    def on_eval_triggered(self):
        if not self.enabled or self._transition_timestep is None:
            return
        if self._post_target_eval_index < len(self.post_target_eval_offsets):
            self._post_target_eval_index += 1

    def describe_phase(self):
        if not self.enabled:
            return "fixed"
        if self._transition_timestep is None:
            return "early"
        if self._post_target_eval_index < len(self.post_target_eval_offsets):
            return "stability"
        return "late"


class EarlyStopOnTSR:
    def __init__(self, config):
        config = config or {}
        self.enabled = bool(config)
        self.target_tsr = float(config.get("target_tsr", 1.0))
        self.consecutive_target_evals = int(config.get("consecutive_target_evals", 1))
        self._target_streak = 0

    def on_eval_result(self, tsr):
        if not self.enabled:
            return False
        if tsr >= self.target_tsr:
            self._target_streak += 1
        else:
            self._target_streak = 0
        return self._target_streak >= self.consecutive_target_evals


def _should_force_final_eval(model, current_timestep, last_eval_timestep, already_forced=False):
    total_timesteps = getattr(model, "_total_timesteps", None)
    if total_timesteps is None:
        return False
    return (
        (not already_forced)
        and current_timestep >= int(total_timesteps)
        and current_timestep != last_eval_timestep
    )

def _sanitize_rule_name(rule_name):
    return str(rule_name).replace(" ", "_")

def _load_task_rule_names(rule_yaml_file):
    with open(rule_yaml_file, "r") as handle:
        data = yaml.safe_load(handle) or {}
    return [rule["name"] for rule in data.get("Rules", {}).get("Task", []) if "name" in rule]

def _eval_csv_fieldnames(row_dict, failure_rule_names):
    base_fieldnames = [
        "exp_name",
        "assignment_name",
        "model_slot_mapping",
        "permutation_index",
        "method",
        "difficulty",
        "seed",
        "split",
        "eval_index",
        "timestep",
        "wall_clock_time_sec",
        "best_tsr_so_far",
        "num_eval_episodes",
        "tsr",
        "mean_reward",
        "failure_rate",
        "timeout_rate",
        "mean_episode_length",
        "num_rl_agents",
        "mean_agent_tsr",
        "joint_tsr",
        "joint_failure_rate",
        "joint_timeout_rate",
        "episodes_all_k_success",
        "episodes_joint_failure",
        "episodes_joint_timeout",
        "mean_completed_rl_agents",
        "mean_completed_rl_fraction",
        "mean_completed_rl_agents_before_failure",
        "mean_completed_rl_agents_before_timeout",
        "action_slow_count",
        "action_normal_count",
        "action_fast_count",
        "action_stop_count",
        "mean_decision_succ",
        "mean_base_policy_safe_prob",
        "mean_shielded_policy_safe_prob",
        "mean_safety_gain",
        "shield_intervention_rate",
        "mean_policy_kl_base_to_shielded",
        "mean_l1_shift_base_to_shielded",
        "hazard_step_rate",
        "shield_forced_stop_rate",
        "base_action_slow_count",
        "base_action_normal_count",
        "base_action_fast_count",
        "base_action_stop_count",
        "shielded_action_slow_count",
        "shielded_action_normal_count",
        "shielded_action_fast_count",
        "shielded_action_stop_count",
        "mean_safe_prob_action_slow",
        "mean_safe_prob_action_normal",
        "mean_safe_prob_action_fast",
        "mean_safe_prob_action_stop",
        "rollout_failures_since_last_eval",
        "rollout_timeouts_since_last_eval",
        "rollout_successes_since_last_eval",
        "goal_completion_rate",
        "mean_agent_goal_completion_rate",
        "failure_rate_per_goal",
        "failure_rate_per_1k_steps",
        "unfinished_goal_rate",
        "completed_goals",
        "failed_goals",
        "active_goals_at_budget",
        "deployment_count",
        "failure_reset_count",
        "resolved_goal_attempts",
        "resolved_goal_quota_per_agent",
        "micro_rollout_tsr",
        "micro_rollout_failure_rate",
        "macro_rollout_tsr",
        "macro_rollout_failure_rate",
        "mean_steps_per_resolved_goal_attempt",
    ]
    dynamic_fields = sorted(
        [
            k for k in row_dict.keys()
            if k not in base_fieldnames
            and not k.startswith("decision_succ_action_")
            and not k.startswith("fail_rl_agent_")
            and not k.startswith("fail_rule_")
        ]
    )
    decision_fields = sorted([k for k in row_dict.keys() if k.startswith("decision_succ_action_")])
    fail_rl_agent_fields = sorted([k for k in row_dict.keys() if k.startswith("fail_rl_agent_")])
    fail_rule_fields = ["fail_rule_{}".format(_sanitize_rule_name(rule_name)) for rule_name in failure_rule_names]
    return base_fieldnames + dynamic_fields + decision_fields + fail_rl_agent_fields + fail_rule_fields

def append_eval_csv_row(csv_path, row_dict, failure_rule_names):
    file_exists = os.path.isfile(csv_path)
    fieldnames = _eval_csv_fieldnames(row_dict, failure_rule_names)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def _resolve_results_dir(results_root, exp_name, fallback_root):
    if results_root is None:
        results_dir = fallback_root
    else:
        results_dir = os.path.join(results_root, exp_name)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def make_env(simulation_config, episode_cache=None, return_cache=False): 
    # Unpack arguments from simulation_config and pass them to CityLoader
    city, cached_observation = CityLoader.from_yaml(**simulation_config, episode_cache=episode_cache)
    if simulation_config.get("rl_agent", {}).get("agent_names"):
        env = SharedPolicyMultiAgentVecEnv(city)
    else:
        env = GymCityWrapper(city)
    if return_cache: 
        return env, cached_observation
    else:
        return env

class EvalCheckpointCallback(CheckpointCallback):
    def __init__(self, exp_name, simulation_config, episode_data, eval_actions, eval_freq=50000, method=None, difficulty=None, seed=None, split="val", adaptive_eval_schedule=None, adaptive_save_schedule=None, validation_debug=None, early_stop=None, results_root=None, *args, **kwargs):
        super(EvalCheckpointCallback, self).__init__(*args, **kwargs)
        self.eval_freq = eval_freq
        self.exp_name = exp_name
        self.method = method
        self.difficulty = difficulty
        self.seed = seed
        self.split = split
        self.best_mean_reward = -np.inf
        self.best_tsr_so_far = 0.0
        self.simulation_config = simulation_config
        with open(episode_data, "rb") as f:
            self.episode_data = pkl.load(f)
        self.eval_actions = eval_actions
        self.results_dir = _resolve_results_dir(results_root, self.exp_name, self.save_path)
        self.eval_csv_path = os.path.join(self.results_dir, "{}_eval_metrics.csv".format(self.exp_name))
        self._last_eval_timestep = 0
        self._last_save_timestep = 0
        self.failure_rule_names = _load_task_rule_names(self.simulation_config["rule_yaml_file"])
        self._rollout_failures_since_last_eval = 0
        self._rollout_timeouts_since_last_eval = 0
        self._rollout_successes_since_last_eval = 0
        self._rollout_stop_needed_steps_since_last_eval = 0
        self._rollout_agent_steps_since_last_eval = 0
        self._eval_index = 0
        self._start_time = time.time()
        self._forced_final_eval_done = False
        self._adaptive_eval_schedule = AdaptiveEvalSchedule(adaptive_eval_schedule, self.eval_freq)
        self._adaptive_save_schedule = AdaptiveEvalSchedule(adaptive_save_schedule, self.save_freq)
        self.validation_debug = validation_debug or {}
        self._early_stop = EarlyStopOnTSR(early_stop)

    def _append_eval_csv_row(self, row_dict):
        append_eval_csv_row(self.eval_csv_path, row_dict, self.failure_rule_names)

    def _on_step(self) -> bool:
        learning_starts = getattr(self.model, "learning_starts", 0)
        current_timestep = self.model.num_timesteps
        infos = self.locals.get("infos")
        if infos is not None:
            for info_idx, info in enumerate(infos):
                if "rl_agent_layer_id" in info:
                    self._rollout_agent_steps_since_last_eval += 1
                if info.get("stop_needed", False):
                    self._rollout_stop_needed_steps_since_last_eval += 1
                joint_failure = info.get("joint_failure", None)
                if joint_failure is not None:
                    if info_idx == 0:
                        if info.get("joint_timeout", False):
                            self._rollout_timeouts_since_last_eval += 1
                        elif joint_failure:
                            self._rollout_failures_since_last_eval += 1
                else:
                    if info.get("overtime", False):
                        self._rollout_timeouts_since_last_eval += 1
                    elif info.get("Fail", [False])[0]:
                        self._rollout_failures_since_last_eval += 1
                if info.get("is_success", False) or info.get("success", False):
                    self._rollout_successes_since_last_eval += 1
        if (
            self._adaptive_save_schedule.should_eval(current_timestep, self._last_save_timestep)
            and (current_timestep > learning_starts)
        ):
            self._last_save_timestep = current_timestep
            self._adaptive_save_schedule.on_eval_triggered()
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(model_path)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

            if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                # If model has a replay buffer, save it too
                replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
                self.model.save_replay_buffer(replay_buffer_path)  # type: ignore[attr-defined]
                if self.verbose > 1:
                    print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

            if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                # Save the VecNormalize statistics
                vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
                self.model.get_vec_normalize_env().save(vec_normalize_path)  # type: ignore[union-attr]
                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")

        # Perform evaluation at specified intervals
        force_final_eval = _should_force_final_eval(
            self.model,
            current_timestep,
            self._last_eval_timestep,
            already_forced=self._forced_final_eval_done,
        )
        if self._adaptive_eval_schedule.should_eval(current_timestep, self._last_eval_timestep) or force_final_eval:
            self._last_eval_timestep = current_timestep
            self._eval_index += 1
            self._adaptive_eval_schedule.on_eval_triggered()
            elapsed_time_sec = time.time() - self._start_time
            rewards_list = []
            success = []
            failures = []
            timeouts = []
            episode_lengths = []
            decision_step = {}
            succ_decision = {}
            action_hist = {0: 0, 1: 0, 2: 0, 3: 0}
            base_action_hist = {0: 0, 1: 0, 2: 0, 3: 0}
            shielded_action_hist = {0: 0, 1: 0, 2: 0, 3: 0}
            fail_rule_counts = {}
            fail_rl_agent_counts = {}
            mean_agent_tsr_list = []
            completed_rl_agents_list = []
            completed_rl_agents_before_failure = []
            completed_rl_agents_before_timeout = []
            per_agent_eval_stats = {}
            num_rl_agents_value = 1
            safe_prob_sums = np.zeros(4, dtype=np.float64)
            base_safe_prob_sum = 0.0
            shielded_safe_prob_sum = 0.0
            intervention_count = 0.0
            kl_sum = 0.0
            l1_sum = 0.0
            hazard_count = 0.0
            forced_stop_count = 0.0
            plpg_step_count = 0
            for action, id in self.eval_actions.items():
                decision_step[id] = 0
                succ_decision[id] = 0
            for ts in list(self.episode_data.keys()):  # Number of episodes for evaluation
                logger.info("Evaluating episode {}...".format(ts))
                episode_cache = self.episode_data[ts]
                if "label_info" in episode_cache:
                    logger.info("Episode label: {}".format(episode_cache["label_info"]))
                eval_env = make_env(self.simulation_config, episode_cache, False)
                is_multi_agent = hasattr(eval_env, "rl_agents")
                debug_enabled = bool(self.validation_debug.get("enabled", False))
                debug_episode = int(self.validation_debug.get("episode_index", -1))
                debug_max_steps = int(self.validation_debug.get("max_steps", 25))
                debug_this_episode = debug_enabled and int(ts) == debug_episode
                debug_pred_grounding_index = getattr(eval_env, "pred_grounding_index", None) if debug_this_episode else None
                if is_multi_agent:
                    max_steps = eval_env.horizon
                else:
                    max_steps = episode_cache["label_info"]["oracle_step"] * 2
                obs = eval_env.init()
                episode_rewards = 0
                step = 0
                local_decision_step = {}
                local_succ_decision = {}
                for acc, id in self.eval_actions.items():
                    local_decision_step[id] = 0
                    local_succ_decision[id] = 1
                done = False
                while (not done) and (step < max_steps):
                    oracle_action = eval_env.expert_action if hasattr(eval_env, "expert_action") else None
                    plpg_info = None
                    if hasattr(self.model, "predict_with_plpg_info"):
                        action, _states, plpg_info = self.model.predict_with_plpg_info(obs, deterministic=True)
                    else:
                        action, _states = self.model.predict(obs, deterministic=True)
                    action_array = np.atleast_1d(action)
                    for action_item in action_array:
                        action_hist[int(action_item)] = action_hist.get(int(action_item), 0) + 1
                    if plpg_info is not None:
                        base_actions = np.atleast_1d(plpg_info["base_action"])
                        shielded_actions = np.atleast_1d(plpg_info["shielded_action"])
                        for base_action in base_actions:
                            base_action_hist[int(base_action)] += 1
                        for shielded_action in shielded_actions:
                            shielded_action_hist[int(shielded_action)] += 1
                        safe_prob_sums += np.asarray(plpg_info["safety_probs"], dtype=np.float64).sum(axis=0)
                        base_safe_prob_sum += float(np.asarray(plpg_info["base_policy_safe_prob"], dtype=np.float64).sum())
                        shielded_safe_prob_sum += float(np.asarray(plpg_info["shielded_policy_safe_prob"], dtype=np.float64).sum())
                        intervention_count += float(np.asarray(plpg_info["intervened"], dtype=np.float64).sum())
                        kl_sum += float(np.asarray(plpg_info["kl_base_to_shielded"], dtype=np.float64).sum())
                        l1_sum += float(np.asarray(plpg_info["l1_shift_base_to_shielded"], dtype=np.float64).sum())
                        hazard_count += float(np.asarray(plpg_info["hazard"], dtype=np.float64).sum())
                        forced_stop_count += float(np.asarray(plpg_info["forced_stop"], dtype=np.float64).sum())
                        plpg_step_count += int(np.asarray(plpg_info["base_policy_safe_prob"]).shape[0]) if np.asarray(plpg_info["base_policy_safe_prob"]).ndim > 0 else 1
                        if debug_this_episode and step < debug_max_steps:
                            fact_snapshots = _extract_easy_debug_facts(obs, debug_pred_grounding_index)
                            base_probs = np.asarray(plpg_info["base_probs"], dtype=np.float64)
                            shielded_probs = np.asarray(plpg_info["shielded_probs"], dtype=np.float64)
                            hazards = np.asarray(plpg_info["hazard"], dtype=np.float64).reshape(-1)
                            interventions = np.asarray(plpg_info["intervened"], dtype=np.float64).reshape(-1)
                            forced_stops = np.asarray(plpg_info["forced_stop"], dtype=np.float64).reshape(-1)
                            for agent_idx in range(base_probs.shape[0]):
                                facts = fact_snapshots[agent_idx] if agent_idx < len(fact_snapshots) else {}
                                logger.info(
                                    "VAL_DEBUG episode=%s step=%s agent=%s oracle=%s base_action=%s shielded_action=%s hazard=%.3f intervened=%.3f forced_stop=%.3f base_probs=%s shielded_probs=%s facts=%s",
                                    ts,
                                    step,
                                    agent_idx,
                                    oracle_action,
                                    ACTION_ID_TO_NAME.get(int(base_actions[agent_idx]), int(base_actions[agent_idx])),
                                    ACTION_ID_TO_NAME.get(int(shielded_actions[agent_idx]), int(shielded_actions[agent_idx])),
                                    float(hazards[agent_idx]) if agent_idx < hazards.shape[0] else 0.0,
                                    float(interventions[agent_idx]) if agent_idx < interventions.shape[0] else 0.0,
                                    float(forced_stops[agent_idx]) if agent_idx < forced_stops.shape[0] else 0.0,
                                    _format_prob_vector(base_probs[agent_idx]),
                                    _format_prob_vector(shielded_probs[agent_idx]),
                                    facts,
                                )
                    chosen_action_cmp = int(action_array[0])
                    if oracle_action is not None and oracle_action in local_decision_step.keys():
                        local_decision_step[oracle_action] = 1
                        if chosen_action_cmp != oracle_action:
                            local_succ_decision[oracle_action] = 0
                    env_action = action if is_multi_agent else int(action)
                    obs, reward, done, info = _unpack_step_result(eval_env.step(env_action))
                    if is_multi_agent:
                        done = bool(np.any(done))
                    if is_multi_agent:
                        reward_value = float(np.mean(reward))
                        joint_info = info[0]
                        num_rl_agents_value = int(joint_info.get("num_rl_agents", num_rl_agents_value))
                        if joint_info.get("joint_failure", False):
                            episode_rewards += reward_value
                            break
                        episode_rewards += reward_value
                    else:
                        if info["Fail"][0]:
                            episode_rewards += reward
                            break
                        episode_rewards += reward
                    step += 1
                if is_multi_agent:
                    joint_info = info[0]
                    completed_rl_agents = int(round(float(joint_info.get("mean_agent_success", 0.0)) * int(joint_info.get("num_rl_agents", num_rl_agents_value))))
                    completed_rl_agents_list.append(completed_rl_agents)
                    _update_per_agent_episode_stats(
                        per_agent_eval_stats,
                        info,
                        joint_fail_agent_layer_ids=joint_info.get("joint_fail_agent_layer_ids", []),
                        joint_timeout=bool(joint_info.get("joint_timeout", False)),
                    )
                    if joint_info.get("joint_is_success", False):
                        logger.info("Episode {} success.".format(ts))
                        success.append(1)
                        failures.append(0)
                        timeouts.append(1 if joint_info.get("joint_timeout", False) else 0)
                    else:
                        logger.info("Episode {} failed.".format(ts))
                        success.append(0)
                        failures.append(1 if joint_info.get("joint_failure", False) else 0)
                        timeouts.append(1 if joint_info.get("joint_timeout", False) else 0)
                        if joint_info.get("joint_failure", False):
                            completed_rl_agents_before_failure.append(completed_rl_agents)
                        if joint_info.get("joint_timeout", False):
                            completed_rl_agents_before_timeout.append(completed_rl_agents)
                    mean_agent_tsr_list.append(float(joint_info.get("mean_agent_success", 0.0)))
                    for layer_id in joint_info.get("joint_fail_agent_layer_ids", []):
                        key = "fail_rl_agent_{}".format(layer_id)
                        fail_rl_agent_counts[key] = fail_rl_agent_counts.get(key, 0) + 1
                elif info["is_success"]:
                    logger.info("Episode {} success.".format(ts))
                    success.append(1)
                    failures.append(0)
                    timeouts.append(1 if info.get("overtime", False) else 0)
                else:
                    logger.info("Episode {} failed.".format(ts))
                    success.append(0)
                    failures.append(1 if info["Fail"][0] else 0)
                    timeouts.append(1 if info.get("overtime", False) else 0)
                    for rule_name in info.get("FailRuleNames", [[]])[0]:
                        fail_rule_counts[rule_name] = fail_rule_counts.get(rule_name, 0) + 1
                if step >= max_steps:
                    episode_rewards -= 3
                    timeouts[-1] = 1
                episode_lengths.append(step)
                for acc, id in self.eval_actions.items():
                    if local_decision_step[id] == 0:
                        local_succ_decision[id] = 0
                    decision_step[id] += local_decision_step[id]
                    succ_decision[id] += local_succ_decision[id]
                rewards_list.append(episode_rewards)
                logger.info("Episode {} achieved a score of {}".format(ts, episode_rewards))
                logger.info("Episode {} Success: {}".format(ts, success[-1]))
                logger.info("Episode {} Decision Step: {}".format(ts, local_decision_step))
                logger.info("Episode {} Success Decision: {}".format(ts, local_succ_decision))


            mean_reward = np.mean(rewards_list)
            sr = np.mean(success)
            self.best_tsr_so_far = max(self.best_tsr_so_far, sr)
            failure_rate = np.mean(failures)
            timeout_rate = np.mean(timeouts)
            mean_length = np.mean(episode_lengths)
            mSuccD, aSuccD, SuccDAct = cal_step_metric(decision_step, succ_decision)
            mean_agent_tsr = np.mean(mean_agent_tsr_list) if len(mean_agent_tsr_list) > 0 else sr
            mean_completed_rl_agents = float(np.mean(completed_rl_agents_list)) if len(completed_rl_agents_list) > 0 else float(sr * num_rl_agents_value)
            mean_completed_rl_fraction = (mean_completed_rl_agents / max(num_rl_agents_value, 1)) if num_rl_agents_value > 0 else 0.0
            mean_completed_rl_agents_before_failure = float(np.mean(completed_rl_agents_before_failure)) if len(completed_rl_agents_before_failure) > 0 else 0.0
            mean_completed_rl_agents_before_timeout = float(np.mean(completed_rl_agents_before_timeout)) if len(completed_rl_agents_before_timeout) > 0 else 0.0
            episodes_all_k_success = int(np.sum(success))
            episodes_joint_failure = int(np.sum(failures))
            episodes_joint_timeout = int(np.sum(timeouts))
            logger.info(
                f"Timestep: {current_timestep} - Elapsed Time (s): {elapsed_time_sec:.2f} - Success Rate: {sr} - Mean Reward: {mean_reward} \n"
            )
            logger.info("Failure Rate: {} - Timeout Rate: {} - Mean Episode Length: {}".format(failure_rate, timeout_rate, mean_length))
            logger.info("Mean Decision Succ: {}".format(mSuccD))
            logger.info("Average Decision Succ: {}".format(aSuccD))
            logger.info("Decision Succ for each action: {}".format(SuccDAct))
            logger.info("Action Histogram: {}".format(action_hist))
            logger.info("Failure Rule Counts: {}".format(fail_rule_counts))
            logger.info(
                "Training Rollout Since Last Eval: failures={} timeouts={} successes={}".format(
                    self._rollout_failures_since_last_eval,
                    self._rollout_timeouts_since_last_eval,
                    self._rollout_successes_since_last_eval,
                )
            )
            logger.info(
                "Training Stop-Demand Since Last Eval: stop_needed_steps={} agent_steps={} required_stop_rate={}".format(
                    self._rollout_stop_needed_steps_since_last_eval,
                    self._rollout_agent_steps_since_last_eval,
                    (self._rollout_stop_needed_steps_since_last_eval / self._rollout_agent_steps_since_last_eval)
                    if self._rollout_agent_steps_since_last_eval > 0 else 0.0,
                )
            )
            logger.info("Eval Schedule Phase: {}".format(self._adaptive_eval_schedule.describe_phase()))
            if plpg_step_count > 0:
                logger.info(
                    "PLPG Metrics: base_safe={} shielded_safe={} gain={} intervention_rate={} kl={} l1={} hazard_rate={} forced_stop_rate={}".format(
                        base_safe_prob_sum / plpg_step_count,
                        shielded_safe_prob_sum / plpg_step_count,
                        (shielded_safe_prob_sum - base_safe_prob_sum) / plpg_step_count,
                        intervention_count / plpg_step_count,
                        kl_sum / plpg_step_count,
                        l1_sum / plpg_step_count,
                        hazard_count / plpg_step_count,
                        forced_stop_count / max(hazard_count, 1.0),
                    )
                )

            # Log the mean reward
            with open(os.path.join(self.save_path, "{}_eval_rewards.txt".format(self.exp_name)), "a") as file:
                file.write(
                    f"Timestep: {current_timestep} - Elapsed Time (s): {elapsed_time_sec:.2f} - Success Rate: {sr} - Mean Reward: {mean_reward} \n"
                )
                file.write("Failure Rate: {} - Timeout Rate: {} - Mean Episode Length: {}\n".format(failure_rate, timeout_rate, mean_length))
                file.write("Mean Decision Succ: {}\n".format(mSuccD))
                file.write("Average Decision Succ: {}\n".format(aSuccD))
                file.write("Decision Succ for each action: {}\n".format(SuccDAct))
                file.write("Action Histogram: {}\n".format(action_hist))
                file.write("Failure Rule Counts: {}\n".format(fail_rule_counts))
                file.write(
                    "Training Rollout Since Last Eval: failures={} timeouts={} successes={}\n".format(
                        self._rollout_failures_since_last_eval,
                        self._rollout_timeouts_since_last_eval,
                        self._rollout_successes_since_last_eval,
                    )
                )
                file.write(
                    "Training Stop-Demand Since Last Eval: stop_needed_steps={} agent_steps={} required_stop_rate={}\n".format(
                        self._rollout_stop_needed_steps_since_last_eval,
                        self._rollout_agent_steps_since_last_eval,
                        (self._rollout_stop_needed_steps_since_last_eval / self._rollout_agent_steps_since_last_eval)
                        if self._rollout_agent_steps_since_last_eval > 0 else 0.0,
                    )
                )
                file.write("Eval Schedule Phase: {}\n".format(self._adaptive_eval_schedule.describe_phase()))
                if plpg_step_count > 0:
                    file.write(
                        "PLPG Metrics: base_safe={} shielded_safe={} gain={} intervention_rate={} kl={} l1={} hazard_rate={} forced_stop_rate={}\n".format(
                            base_safe_prob_sum / plpg_step_count,
                            shielded_safe_prob_sum / plpg_step_count,
                            (shielded_safe_prob_sum - base_safe_prob_sum) / plpg_step_count,
                            intervention_count / plpg_step_count,
                            kl_sum / plpg_step_count,
                            l1_sum / plpg_step_count,
                            hazard_count / plpg_step_count,
                            forced_stop_count / max(hazard_count, 1.0),
                        )
                    )

            csv_row = {
                "exp_name": self.exp_name,
                "method": self.method or getattr(self.model, "__class__", type(self.model)).__name__,
                "difficulty": self.difficulty or "",
                "seed": self.seed if self.seed is not None else "",
                "split": self.split,
                "eval_index": self._eval_index,
                "timestep": current_timestep,
                "wall_clock_time_sec": elapsed_time_sec,
                "best_tsr_so_far": self.best_tsr_so_far,
                "num_eval_episodes": len(rewards_list),
                "tsr": sr,
                "mean_reward": mean_reward,
                "failure_rate": failure_rate,
                "timeout_rate": timeout_rate,
                "mean_episode_length": mean_length,
                "num_rl_agents": num_rl_agents_value,
                "mean_agent_tsr": mean_agent_tsr,
                "joint_tsr": sr,
                "joint_failure_rate": failure_rate,
                "joint_timeout_rate": timeout_rate,
                "episodes_all_k_success": episodes_all_k_success,
                "episodes_joint_failure": episodes_joint_failure,
                "episodes_joint_timeout": episodes_joint_timeout,
                "mean_completed_rl_agents": mean_completed_rl_agents,
                "mean_completed_rl_fraction": mean_completed_rl_fraction,
                "mean_completed_rl_agents_before_failure": mean_completed_rl_agents_before_failure,
                "mean_completed_rl_agents_before_timeout": mean_completed_rl_agents_before_timeout,
                "mean_decision_succ": mSuccD,
                "rollout_failures_since_last_eval": self._rollout_failures_since_last_eval,
                "rollout_timeouts_since_last_eval": self._rollout_timeouts_since_last_eval,
                "rollout_successes_since_last_eval": self._rollout_successes_since_last_eval,
                "rollout_stop_needed_steps_since_last_eval": self._rollout_stop_needed_steps_since_last_eval,
                "rollout_agent_steps_since_last_eval": self._rollout_agent_steps_since_last_eval,
                "rollout_required_stop_rate_since_last_eval": (
                    self._rollout_stop_needed_steps_since_last_eval / self._rollout_agent_steps_since_last_eval
                ) if self._rollout_agent_steps_since_last_eval > 0 else 0.0,
                "mean_base_policy_safe_prob": (base_safe_prob_sum / plpg_step_count) if plpg_step_count > 0 else 0.0,
                "mean_shielded_policy_safe_prob": (shielded_safe_prob_sum / plpg_step_count) if plpg_step_count > 0 else 0.0,
                "mean_safety_gain": ((shielded_safe_prob_sum - base_safe_prob_sum) / plpg_step_count) if plpg_step_count > 0 else 0.0,
                "shield_intervention_rate": (intervention_count / plpg_step_count) if plpg_step_count > 0 else 0.0,
                "mean_policy_kl_base_to_shielded": (kl_sum / plpg_step_count) if plpg_step_count > 0 else 0.0,
                "mean_l1_shift_base_to_shielded": (l1_sum / plpg_step_count) if plpg_step_count > 0 else 0.0,
                "hazard_step_rate": (hazard_count / plpg_step_count) if plpg_step_count > 0 else 0.0,
                "shield_forced_stop_rate": (forced_stop_count / max(hazard_count, 1.0)) if plpg_step_count > 0 else 0.0,
            }
            for action_id, action_name in ACTION_ID_TO_NAME.items():
                csv_row[f"action_{action_name}_count"] = action_hist.get(action_id, 0)
                csv_row[f"base_action_{action_name}_count"] = base_action_hist.get(action_id, 0)
                csv_row[f"shielded_action_{action_name}_count"] = shielded_action_hist.get(action_id, 0)
                csv_row[f"mean_safe_prob_action_{action_name}"] = (
                    safe_prob_sums[action_id] / plpg_step_count
                ) if plpg_step_count > 0 else 0.0
            for action_id, value in SuccDAct.items():
                action_name = ACTION_ID_TO_NAME.get(action_id, str(action_id))
                csv_row["decision_succ_action_{}".format(action_name)] = value
            for key, value in fail_rl_agent_counts.items():
                csv_row[key] = value
            csv_row.update(_build_per_agent_eval_row(per_agent_eval_stats))
            for rule_name in self.failure_rule_names:
                csv_row["fail_rule_{}".format(_sanitize_rule_name(rule_name))] = fail_rule_counts.get(rule_name, 0)
            self._append_eval_csv_row(csv_row)
            self._adaptive_eval_schedule.on_eval_result(current_timestep, sr)
            self._adaptive_save_schedule.on_eval_result(current_timestep, sr)
            should_stop_early = self._early_stop.on_eval_result(sr)
            if force_final_eval:
                self._forced_final_eval_done = True
            self._rollout_failures_since_last_eval = 0
            self._rollout_timeouts_since_last_eval = 0
            self._rollout_successes_since_last_eval = 0
            self._rollout_stop_needed_steps_since_last_eval = 0
            self._rollout_agent_steps_since_last_eval = 0

            # Update the best model if current mean reward is better
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save("{}/best_model.zip".format(self.save_path))

            if should_stop_early:
                logger.info(
                    "Early stopping triggered at timestep %s after validation TSR reached %.4f.",
                    current_timestep,
                    sr,
                )
                return False

        return True

class DreamerEvalCheckpointCallback(CheckpointCallback):
    def __init__(self, exp_name, simulation_config, episode_data, eval_actions, eval_freq=50000, method=None, difficulty=None, seed=None, split="val", adaptive_eval_schedule=None, adaptive_save_schedule=None, early_stop=None, results_root=None, *args, **kwargs):
        super(DreamerEvalCheckpointCallback, self).__init__(*args, **kwargs)
        self.eval_freq = eval_freq
        self.exp_name = exp_name
        self.method = method
        self.difficulty = difficulty
        self.seed = seed
        self.split = split
        self.best_mean_reward = -np.inf
        self.best_tsr_so_far = 0.0
        self.simulation_config = simulation_config
        with open(episode_data, "rb") as f:
            self.episode_data = pkl.load(f)
        self.eval_actions = eval_actions
        self.results_dir = _resolve_results_dir(results_root, self.exp_name, self.save_path)
        self.eval_csv_path = os.path.join(self.results_dir, "{}_eval_metrics.csv".format(self.exp_name))
        self._last_eval_timestep = 0
        self._last_save_timestep = 0
        self.failure_rule_names = _load_task_rule_names(self.simulation_config["rule_yaml_file"])
        self._rollout_failures_since_last_eval = 0
        self._rollout_timeouts_since_last_eval = 0
        self._rollout_successes_since_last_eval = 0
        self._eval_index = 0
        self._start_time = time.time()
        self._forced_final_eval_done = False
        self._adaptive_eval_schedule = AdaptiveEvalSchedule(adaptive_eval_schedule, self.eval_freq)
        self._adaptive_save_schedule = AdaptiveEvalSchedule(adaptive_save_schedule, self.save_freq)
        self._early_stop = EarlyStopOnTSR(early_stop)

    def _append_eval_csv_row(self, row_dict):
        append_eval_csv_row(self.eval_csv_path, row_dict, self.failure_rule_names)

    def _on_step(self) -> bool:
        current_timestep = self.model.num_timesteps
        infos = self.locals.get("infos")
        if infos is not None:
            for info_idx, info in enumerate(infos):
                joint_failure = info.get("joint_failure", None)
                if joint_failure is not None:
                    if info_idx == 0:
                        if info.get("joint_timeout", False):
                            self._rollout_timeouts_since_last_eval += 1
                        elif joint_failure:
                            self._rollout_failures_since_last_eval += 1
                else:
                    if info.get("overtime", False):
                        self._rollout_timeouts_since_last_eval += 1
                    elif info.get("Fail", [False])[0]:
                        self._rollout_failures_since_last_eval += 1
                if info.get("is_success", False) or info.get("success", False):
                    self._rollout_successes_since_last_eval += 1
        if self._adaptive_save_schedule.should_eval(current_timestep, self._last_save_timestep):
            self._last_save_timestep = current_timestep
            self._adaptive_save_schedule.on_eval_triggered()
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(model_path)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

            if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                # If model has a replay buffer, save it too
                replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
                self.model.save_replay_buffer(replay_buffer_path)  # type: ignore[attr-defined]
                if self.verbose > 1:
                    print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

            if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                # Save the VecNormalize statistics
                vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
                self.model.get_vec_normalize_env().save(vec_normalize_path)  # type: ignore[union-attr]
                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")

        # Perform evaluation at specified intervals
        force_final_eval = _should_force_final_eval(
            self.model,
            current_timestep,
            self._last_eval_timestep,
            already_forced=self._forced_final_eval_done,
        )
        if self._adaptive_eval_schedule.should_eval(current_timestep, self._last_eval_timestep) or force_final_eval:
            self._last_eval_timestep = current_timestep
            self._eval_index += 1
            self._adaptive_eval_schedule.on_eval_triggered()
            elapsed_time_sec = time.time() - self._start_time
            rewards_list = []
            success = []
            failures = []
            timeouts = []
            episode_lengths = []
            decision_step = {}
            succ_decision = {}
            action_hist = {0: 0, 1: 0, 2: 0, 3: 0}
            fail_rule_counts = {}
            fail_rl_agent_counts = {}
            mean_agent_tsr_list = []
            completed_rl_agents_list = []
            completed_rl_agents_before_failure = []
            completed_rl_agents_before_timeout = []
            per_agent_eval_stats = {}
            num_rl_agents_value = 1
            for action, id in self.eval_actions.items():
                decision_step[id] = 0
                succ_decision[id] = 0
            for ts in list(self.episode_data.keys()):  # Number of episodes for evaluation
                logger.info("Evaluating episode {}...".format(ts))
                episode_cache = self.episode_data[ts]
                if "label_info" in episode_cache:
                    logger.info("Episode label: {}".format(episode_cache["label_info"]))
                eval_env = make_env(self.simulation_config, episode_cache, False)
                is_multi_agent = hasattr(eval_env, "rl_agents")
                if is_multi_agent:
                    max_steps = eval_env.horizon
                else:
                    max_steps = episode_cache["label_info"]["oracle_step"] * 2
                obs = eval_env.init()
                episode_rewards = 0
                step = 0
                prev_rssmstate = self.model.policy.RSSM._init_rssm_state(1)
                prev_action = torch.zeros(1, self.model.action_size).to(self.model.device)
                local_decision_step = {}
                local_succ_decision = {}
                for acc, id in self.eval_actions.items():
                    local_decision_step[id] = 0
                    local_succ_decision[id] = 1
                done = False
                while (not done) and (step < max_steps):
                    oracle_action = eval_env.expert_action
                    with torch.no_grad():
                        embed = self.model.policy.ObsEncoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.model.device))    
                        _, posterior_rssm_state = self.model.policy.RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                        model_state = self.model.policy.RSSM.get_model_state(posterior_rssm_state)
                        action, _ = self.model.policy.ActionModel(model_state)
                        prev_rssmstate = posterior_rssm_state
                        prev_action = action
                    env_action = torch.argmax(action, dim=-1).cpu().numpy()
                    action_array = np.atleast_1d(env_action)
                    for action_item in action_array:
                        action_hist[int(action_item)] = action_hist.get(int(action_item), 0) + 1
                    if oracle_action in local_decision_step.keys():
                        local_decision_step[oracle_action] = 1
                        if int(np.atleast_1d(env_action)[0]) != oracle_action:
                            local_succ_decision[oracle_action] = 0
                    obs, reward, done, info = _unpack_step_result(eval_env.step(env_action))
                    if is_multi_agent:
                        done = bool(np.any(done))
                    if is_multi_agent:
                        reward_value = float(np.mean(reward))
                        joint_info = info[0]
                        num_rl_agents_value = int(joint_info.get("num_rl_agents", num_rl_agents_value))
                        if joint_info.get("joint_failure", False):
                            episode_rewards += reward_value
                            break
                        episode_rewards += reward_value
                    else:
                        if info["Fail"][0]:
                            episode_rewards += reward
                            break
                        episode_rewards += reward
                    step += 1
                if is_multi_agent:
                    joint_info = info[0]
                    completed_rl_agents = int(round(float(joint_info.get("mean_agent_success", 0.0)) * int(joint_info.get("num_rl_agents", num_rl_agents_value))))
                    completed_rl_agents_list.append(completed_rl_agents)
                    _update_per_agent_episode_stats(
                        per_agent_eval_stats,
                        info,
                        joint_fail_agent_layer_ids=joint_info.get("joint_fail_agent_layer_ids", []),
                        joint_timeout=bool(joint_info.get("joint_timeout", False)),
                    )
                    if joint_info.get("joint_is_success", False):
                        logger.info("Episode {} success.".format(ts))
                        success.append(1)
                        failures.append(0)
                        timeouts.append(1 if joint_info.get("joint_timeout", False) else 0)
                    else:
                        logger.info("Episode {} failed.".format(ts))
                        success.append(0)
                        failures.append(1 if joint_info.get("joint_failure", False) else 0)
                        timeouts.append(1 if joint_info.get("joint_timeout", False) else 0)
                        if joint_info.get("joint_failure", False):
                            completed_rl_agents_before_failure.append(completed_rl_agents)
                        if joint_info.get("joint_timeout", False):
                            completed_rl_agents_before_timeout.append(completed_rl_agents)
                    mean_agent_tsr_list.append(float(joint_info.get("mean_agent_success", 0.0)))
                    for layer_id in joint_info.get("joint_fail_agent_layer_ids", []):
                        key = "fail_rl_agent_{}".format(layer_id)
                        fail_rl_agent_counts[key] = fail_rl_agent_counts.get(key, 0) + 1
                elif info["success"]:
                    logger.info("Episode {} success.".format(ts))
                    success.append(1)
                    failures.append(0)
                    timeouts.append(1 if info.get("overtime", False) else 0)
                else:
                    logger.info("Episode {} failed.".format(ts))
                    success.append(0)
                    failures.append(1 if info["Fail"][0] else 0)
                    timeouts.append(1 if info.get("overtime", False) else 0)
                    for rule_name in info.get("FailRuleNames", [[]])[0]:
                        fail_rule_counts[rule_name] = fail_rule_counts.get(rule_name, 0) + 1
                if step >= max_steps:
                    episode_rewards -= 3
                    timeouts[-1] = 1
                episode_lengths.append(step)
                for acc, id in self.eval_actions.items():
                    if local_decision_step[id] == 0:
                        local_succ_decision[id] = 0
                    decision_step[id] += local_decision_step[id]
                    succ_decision[id] += local_succ_decision[id]
                rewards_list.append(episode_rewards)
                logger.info("Episode {} achieved a score of {}".format(ts, episode_rewards))
                logger.info("Episode {} Success: {}".format(ts, success[-1]))
                logger.info("Episode {} Decision Step: {}".format(ts, local_decision_step))
                logger.info("Episode {} Success Decision: {}".format(ts, local_succ_decision))


            mean_reward = np.mean(rewards_list)
            sr = np.mean(success)
            self.best_tsr_so_far = max(self.best_tsr_so_far, sr)
            failure_rate = np.mean(failures)
            timeout_rate = np.mean(timeouts)
            mean_length = np.mean(episode_lengths)
            mSuccD, aSuccD, SuccDAct = cal_step_metric(decision_step, succ_decision)
            mean_agent_tsr = np.mean(mean_agent_tsr_list) if len(mean_agent_tsr_list) > 0 else sr
            mean_completed_rl_agents = float(np.mean(completed_rl_agents_list)) if len(completed_rl_agents_list) > 0 else float(sr * num_rl_agents_value)
            mean_completed_rl_fraction = (mean_completed_rl_agents / max(num_rl_agents_value, 1)) if num_rl_agents_value > 0 else 0.0
            mean_completed_rl_agents_before_failure = float(np.mean(completed_rl_agents_before_failure)) if len(completed_rl_agents_before_failure) > 0 else 0.0
            mean_completed_rl_agents_before_timeout = float(np.mean(completed_rl_agents_before_timeout)) if len(completed_rl_agents_before_timeout) > 0 else 0.0
            episodes_all_k_success = int(np.sum(success))
            episodes_joint_failure = int(np.sum(failures))
            episodes_joint_timeout = int(np.sum(timeouts))
            logger.info(
                f"Timestep: {current_timestep} - Elapsed Time (s): {elapsed_time_sec:.2f} - Success Rate: {sr} - Mean Reward: {mean_reward} \n"
            )
            logger.info("Failure Rate: {} - Timeout Rate: {} - Mean Episode Length: {}".format(failure_rate, timeout_rate, mean_length))
            logger.info("Mean Decision Succ: {}".format(mSuccD))
            logger.info("Average Decision Succ: {}".format(aSuccD))
            logger.info("Decision Succ for each action: {}".format(SuccDAct))
            logger.info("Action Histogram: {}".format(action_hist))
            logger.info("Failure Rule Counts: {}".format(fail_rule_counts))
            logger.info(
                "Training Rollout Since Last Eval: failures={} timeouts={} successes={}".format(
                    self._rollout_failures_since_last_eval,
                    self._rollout_timeouts_since_last_eval,
                    self._rollout_successes_since_last_eval,
                )
            )
            logger.info("Eval Schedule Phase: {}".format(self._adaptive_eval_schedule.describe_phase()))

            # Log the mean reward
            with open(os.path.join(self.save_path, "{}_eval_rewards.txt".format(self.exp_name)), "a") as file:
                file.write(
                    f"Timestep: {current_timestep} - Elapsed Time (s): {elapsed_time_sec:.2f} - Success Rate: {sr} - Mean Reward: {mean_reward} \n"
                )
                file.write("Failure Rate: {} - Timeout Rate: {} - Mean Episode Length: {}\n".format(failure_rate, timeout_rate, mean_length))
                file.write("Mean Decision Succ: {}\n".format(mSuccD))
                file.write("Average Decision Succ: {}\n".format(aSuccD))
                file.write("Decision Succ for each action: {}\n".format(SuccDAct))
                file.write("Action Histogram: {}\n".format(action_hist))
                file.write("Failure Rule Counts: {}\n".format(fail_rule_counts))
                file.write(
                    "Training Rollout Since Last Eval: failures={} timeouts={} successes={}\n".format(
                        self._rollout_failures_since_last_eval,
                        self._rollout_timeouts_since_last_eval,
                        self._rollout_successes_since_last_eval,
                    )
                )
                file.write("Eval Schedule Phase: {}\n".format(self._adaptive_eval_schedule.describe_phase()))

            csv_row = {
                "exp_name": self.exp_name,
                "method": self.method or getattr(self.model, "__class__", type(self.model)).__name__,
                "difficulty": self.difficulty or "",
                "seed": self.seed if self.seed is not None else "",
                "split": self.split,
                "eval_index": self._eval_index,
                "timestep": current_timestep,
                "wall_clock_time_sec": elapsed_time_sec,
                "best_tsr_so_far": self.best_tsr_so_far,
                "num_eval_episodes": len(rewards_list),
                "tsr": sr,
                "mean_reward": mean_reward,
                "failure_rate": failure_rate,
                "timeout_rate": timeout_rate,
                "mean_episode_length": mean_length,
                "num_rl_agents": num_rl_agents_value,
                "mean_agent_tsr": mean_agent_tsr,
                "joint_tsr": sr,
                "joint_failure_rate": failure_rate,
                "joint_timeout_rate": timeout_rate,
                "episodes_all_k_success": episodes_all_k_success,
                "episodes_joint_failure": episodes_joint_failure,
                "episodes_joint_timeout": episodes_joint_timeout,
                "mean_completed_rl_agents": mean_completed_rl_agents,
                "mean_completed_rl_fraction": mean_completed_rl_fraction,
                "mean_completed_rl_agents_before_failure": mean_completed_rl_agents_before_failure,
                "mean_completed_rl_agents_before_timeout": mean_completed_rl_agents_before_timeout,
                "mean_decision_succ": mSuccD,
                "rollout_failures_since_last_eval": self._rollout_failures_since_last_eval,
                "rollout_timeouts_since_last_eval": self._rollout_timeouts_since_last_eval,
                "rollout_successes_since_last_eval": self._rollout_successes_since_last_eval,
            }
            for action_id, action_name in ACTION_ID_TO_NAME.items():
                csv_row[f"action_{action_name}_count"] = action_hist.get(action_id, 0)
            for action_id, value in SuccDAct.items():
                action_name = ACTION_ID_TO_NAME.get(action_id, str(action_id))
                csv_row["decision_succ_action_{}".format(action_name)] = value
            for key, value in fail_rl_agent_counts.items():
                csv_row[key] = value
            csv_row.update(_build_per_agent_eval_row(per_agent_eval_stats))
            for rule_name in self.failure_rule_names:
                csv_row["fail_rule_{}".format(_sanitize_rule_name(rule_name))] = fail_rule_counts.get(rule_name, 0)
            self._append_eval_csv_row(csv_row)
            self._adaptive_eval_schedule.on_eval_result(current_timestep, sr)
            self._adaptive_save_schedule.on_eval_result(current_timestep, sr)
            should_stop_early = self._early_stop.on_eval_result(sr)
            if force_final_eval:
                self._forced_final_eval_done = True
            self._rollout_failures_since_last_eval = 0
            self._rollout_timeouts_since_last_eval = 0
            self._rollout_successes_since_last_eval = 0

            # Update the best model if current mean reward is better
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save("{}/best_model.zip".format(self.save_path))

            if should_stop_early:
                logger.info(
                    "Early stopping triggered at timestep %s after validation TSR reached %.4f.",
                    current_timestep,
                    sr,
                )
                return False

        return True

def cal_step_metric(decision_step, succ_decision):
    mean_decision_succ = {}
    total_decision = sum(decision_step.values())
    total_decision = max(total_decision, 1)
    total_succ = sum(succ_decision.values())
    for action, num in decision_step.items():
        num = max(num, 1)
        mean_decision_succ[action] = succ_decision[action]/num
    average_decision_succ = sum(mean_decision_succ.values())/len(mean_decision_succ)
    # mean decision succ (over all steps), average decision succ (over all actions), decision succ for each action
    return total_succ/total_decision, average_decision_succ, mean_decision_succ
