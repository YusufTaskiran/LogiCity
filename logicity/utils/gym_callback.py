from stable_baselines3.common.callbacks import CheckpointCallback
from logicity.utils.load import CityLoader
from logicity.utils.gym_wrapper import GymCityWrapper
import numpy as np
import torch
import os
import csv
import logging
import pickle as pkl
import time
logger = logging.getLogger(__name__)

def _extract_action_probs(model, obs):
    policy = getattr(model, "policy", None)
    if policy is None or not hasattr(policy, "get_distribution"):
        return None
    try:
        obs_tensor, _ = policy.obs_to_tensor(obs)
        distribution = policy.get_distribution(obs_tensor)
        dist = getattr(distribution, "distribution", distribution)
        probs = getattr(dist, "probs", None)
        if probs is None:
            return None
        return probs.detach().cpu().numpy()
    except Exception:
        return None

def make_env(simulation_config, episode_cache=None, return_cache=False): 
    # Unpack arguments from simulation_config and pass them to CityLoader
    city, cached_observation = CityLoader.from_yaml(**simulation_config, episode_cache=episode_cache)
    env = GymCityWrapper(city)
    if return_cache: 
        return env, cached_observation
    else:
        return env

class EvalCheckpointCallback(CheckpointCallback):
    def __init__(self, exp_name, simulation_config, episode_data, eval_actions, eval_freq=50000, *args, **kwargs):
        super(EvalCheckpointCallback, self).__init__(*args, **kwargs)
        self.eval_freq = eval_freq
        self.exp_name = exp_name
        self.best_mean_reward = -np.inf
        self.start_time = time.time()
        self.simulation_config = simulation_config
        with open(episode_data, "rb") as f:
            self.episode_data = pkl.load(f)
        self.eval_actions = eval_actions
        self.stop_action_id = int(self.eval_actions.get("Stop", next(iter(self.eval_actions.values()))))
        self.num_actions = int(self.simulation_config.get("rl_agent", {}).get("action_space", 2))
        self.csv_path = os.path.join(self.save_path, f"{self.exp_name}_metrics.csv")
        rl_cfg = self.simulation_config.get("rl_agent", {})
        self.debug_action_probs = bool(rl_cfg.get("debug_action_probs", False))
        self.debug_action_prob_steps = int(rl_cfg.get("debug_action_prob_steps", 5))
        self.debug_shield_snapshot = bool(rl_cfg.get("debug_shield_snapshot", False))
        self.debug_shield_snapshot_steps = int(rl_cfg.get("debug_shield_snapshot_steps", 10))
        self.train_fail = 0
        self.train_timeout = 0
        self.train_success = 0
        self.train_truncated = 0
        self.train_completed = 0
        self._last_eval_train_fail = 0
        self._last_eval_train_timeout = 0
        self._last_eval_train_success = 0
        self._last_eval_train_truncated = 0
        self._last_eval_train_completed = 0

    def _update_training_episode_counters(self):
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if infos is None or dones is None:
            return
        for done, info in zip(dones, infos):
            if not done:
                continue
            self.train_completed += 1
            if info.get("is_success", False):
                self.train_success += 1
            elif info.get("overtime", False):
                self.train_timeout += 1
            elif "Fail" in info and len(info["Fail"]) > 0 and bool(info["Fail"][0]):
                self.train_fail += 1
            else:
                self.train_truncated += 1

    def _append_metrics_csv(self, step, sr, SuccDAct, extra_metrics=None, mean_reward=None):
        os.makedirs(self.save_path, exist_ok=True)
        fieldnames = [
            "step",
            "tsr",
            "dsr",
            "mean_reward",
            "fail",
            "timeout",
            "truncated",
            "policy_stop_count",
            "expert_stop_count",
            "matched_stop_count",
            "policy_action_hist_0",
            "policy_action_hist_1",
            "policy_action_hist_2",
            "policy_action_hist_3",
            "expert_action_hist_0",
            "expert_action_hist_1",
            "expert_action_hist_2",
            "expert_action_hist_3",
            "shield_intervention_count",
            "shield_intervention_rate",
            "elapsed_wall_clock_seconds",
            "seconds_per_1k_timesteps",
            "train_fail",
            "train_timeout",
            "train_success",
            "train_truncated",
            "train_completed",
            "train_fail_since_last_eval",
            "train_timeout_since_last_eval",
            "train_success_since_last_eval",
            "train_truncated_since_last_eval",
            "train_completed_since_last_eval",
        ]
        row = {
            "step": step,
            "tsr": sr,
            "dsr": SuccDAct.get(self.stop_action_id, 0.0),
            "mean_reward": mean_reward,
            "fail": 0,
            "timeout": 0,
            "truncated": 0,
            "policy_stop_count": 0,
            "expert_stop_count": 0,
            "matched_stop_count": 0,
            "policy_action_hist_0": 0,
            "policy_action_hist_1": 0,
            "policy_action_hist_2": 0,
            "policy_action_hist_3": 0,
            "expert_action_hist_0": 0,
            "expert_action_hist_1": 0,
            "expert_action_hist_2": 0,
            "expert_action_hist_3": 0,
            "shield_intervention_count": 0,
            "shield_intervention_rate": 0.0,
            "elapsed_wall_clock_seconds": 0.0,
            "seconds_per_1k_timesteps": 0.0,
            "train_fail": 0,
            "train_timeout": 0,
            "train_success": 0,
            "train_truncated": 0,
            "train_completed": 0,
            "train_fail_since_last_eval": 0,
            "train_timeout_since_last_eval": 0,
            "train_success_since_last_eval": 0,
            "train_truncated_since_last_eval": 0,
            "train_completed_since_last_eval": 0,
        }
        if extra_metrics is not None:
            for key, value in extra_metrics.items():
                row[key] = value
            fieldnames.extend([key for key in extra_metrics.keys() if key not in fieldnames])
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _on_step(self) -> bool:
        self._update_training_episode_counters()
        learning_starts = getattr(self.model, "learning_starts", 0)
        if self.n_calls % self.save_freq == 0 and (self.model.num_timesteps > learning_starts):
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
        if self.n_calls % self.eval_freq == 0:
            if hasattr(self.model, "reset_shield_metrics"):
                self.model.reset_shield_metrics()
            rewards_list = []
            success = []
            decision_step = {}
            succ_decision = {}
            policy_action_hist = {}
            expert_action_hist = {}
            fail_episodes = 0
            timeout_episodes = 0
            truncated_episodes = 0
            episode_lengths = []
            oracle_steps = []
            predicted_stop_total = 0
            expert_stop_total = 0
            matched_stop_total = 0
            failed_before_first_required_stop = 0
            first_eval_episode = list(self.episode_data.keys())[0] if len(self.episode_data) > 0 else None
            for action, id in self.eval_actions.items():
                decision_step[id] = 0
                succ_decision[id] = 0
            for ts in list(self.episode_data.keys()):  # Number of episodes for evaluation
                logger.info("Evaluating episode {}...".format(ts))
                episode_cache = self.episode_data[ts]
                if "label_info" in episode_cache:
                    logger.info("Episode label: {}".format(episode_cache["label_info"]))
                max_steps = episode_cache["label_info"]["oracle_step"] * 2
                oracle_steps.append(int(episode_cache["label_info"]["oracle_step"]))
                eval_env = make_env(self.simulation_config, episode_cache, False)
                obs = eval_env.init()
                episode_rewards = 0
                step = 0
                local_decision_step = {}
                local_succ_decision = {}
                local_policy_action_hist = {}
                local_expert_action_hist = {}
                local_predicted_stop = 0
                local_expert_stop = 0
                local_matched_stop = 0
                saw_required_stop = False
                logged_first_required_stop = False
                failed_reason = "unknown"
                for acc, id in self.eval_actions.items():
                    local_decision_step[id] = 0
                    local_succ_decision[id] = 0
                done = False
                while (not done) and (step < max_steps):
                    oracle_action = eval_env.expert_action
                    local_expert_action_hist[oracle_action] = local_expert_action_hist.get(oracle_action, 0) + 1
                    expert_action_hist[oracle_action] = expert_action_hist.get(oracle_action, 0) + 1
                    action_probs = None
                    if self.debug_action_probs and ts == first_eval_episode and step < self.debug_action_prob_steps:
                        action_probs = _extract_action_probs(self.model, obs)
                    shield_snapshot = None
                    if (
                        self.debug_shield_snapshot
                        and ts == first_eval_episode
                        and step < self.debug_shield_snapshot_steps
                        and hasattr(self.model, "debug_action_snapshot")
                    ):
                        if hasattr(self.model, "set_shield_context_from_env"):
                            self.model.set_shield_context_from_env(eval_env, obs)
                        try:
                            shield_snapshot = self.model.debug_action_snapshot(obs)
                        except Exception as exc:
                            logger.info(
                                "Eval shield snapshot failed episode=%s step=%s error=%s",
                                ts,
                                step,
                                exc,
                            )
                    if hasattr(self.model, "set_shield_context_from_env"):
                        self.model.set_shield_context_from_env(eval_env, obs)
                    action, _states = self.model.predict(obs, deterministic=True)
                    action_int = int(action)
                    local_policy_action_hist[action_int] = local_policy_action_hist.get(action_int, 0) + 1
                    policy_action_hist[action_int] = policy_action_hist.get(action_int, 0) + 1
                    if action_probs is not None:
                        logger.info(
                            "Eval action probs episode=%s step=%s probs=%s action=%s expert=%s",
                            ts,
                            step,
                            np.asarray(action_probs).tolist(),
                            action_int,
                            oracle_action,
                        )
                    if shield_snapshot is not None:
                        logger.info(
                            "Eval shield snapshot episode=%s step=%s safe_bits=%s shield_facts=%s base_probs=%s shielded_probs=%s safe_probs=%s base_safety=%.4f shielded_safety=%.4f action=%s expert=%s",
                            ts,
                            step,
                            {
                                key: shield_snapshot[key]
                                for key in ("IsSafeStep1", "IsSafeStep2", "IsSafeStep3", "IsSafeWait")
                                if key in shield_snapshot
                            },
                            shield_snapshot.get("shield_facts", {}),
                            shield_snapshot.get("base_probs"),
                            shield_snapshot.get("shielded_probs"),
                            shield_snapshot.get("safe_probs"),
                            float(shield_snapshot.get("base_safety_prob", 0.0)),
                            float(shield_snapshot.get("shielded_safety_prob", 0.0)),
                            action_int,
                            oracle_action,
                        )
                    if oracle_action in local_decision_step.keys():
                        saw_required_stop = True
                        local_decision_step[oracle_action] += 1
                        expert_stop_total += 1
                        local_expert_stop += 1
                        if (not logged_first_required_stop) and hasattr(self.model, "debug_action_snapshot"):
                            if hasattr(self.model, "set_shield_context_from_env"):
                                self.model.set_shield_context_from_env(eval_env, obs)
                            try:
                                stop_snapshot = self.model.debug_action_snapshot(obs)
                                logger.info(
                                    "Eval first required-stop snapshot episode=%s step=%s shield_facts=%s base_probs=%s shielded_probs=%s safe_probs=%s base_safety=%.4f shielded_safety=%.4f action=%s expert=%s",
                                    ts,
                                    step,
                                    stop_snapshot.get("shield_facts", {}),
                                    stop_snapshot.get("base_probs"),
                                    stop_snapshot.get("shielded_probs"),
                                    stop_snapshot.get("safe_probs"),
                                    float(stop_snapshot.get("base_safety_prob", 0.0)),
                                    float(stop_snapshot.get("shielded_safety_prob", 0.0)),
                                    None,
                                    oracle_action,
                                )
                            except Exception as exc:
                                logger.info(
                                    "Eval first required-stop snapshot failed episode=%s step=%s error=%s",
                                    ts,
                                    step,
                                    exc,
                                )
                            logged_first_required_stop = True
                        if action_int == oracle_action:
                            local_succ_decision[oracle_action] += 1
                            matched_stop_total += 1
                            local_matched_stop += 1
                    if action_int in local_decision_step.keys():
                        predicted_stop_total += 1
                        local_predicted_stop += 1
                    obs, reward, done, info = eval_env.step(action_int)
                    if info["Fail"][0]:
                        failed_reason = "fail"
                        episode_rewards += reward
                        break
                    episode_rewards += reward
                    step += 1
                episode_lengths.append(step)
                if info["is_success"]:
                    failed_reason = "success"
                    logger.info("Episode {} success.".format(ts))
                    success.append(1)
                else:
                    if info.get("overtime", False):
                        failed_reason = "overtime"
                        timeout_episodes += 1
                    elif failed_reason != "fail":
                        failed_reason = "failed"
                    logger.info("Episode {} failed.".format(ts))
                    success.append(0)
                    if failed_reason == "fail":
                        fail_episodes += 1
                    if (not saw_required_stop) and any(v > 0 for v in local_decision_step.values()):
                        failed_before_first_required_stop += 1
                if step >= max_steps:
                    episode_rewards -= 6
                    truncated_episodes += 1
                for acc, id in self.eval_actions.items():
                    decision_step[id] += local_decision_step[id]
                    succ_decision[id] += local_succ_decision[id]
                rewards_list.append(episode_rewards)
                logger.info("Episode {} achieved a score of {}".format(ts, episode_rewards))
                logger.info("Episode {} Success: {}".format(ts, success[-1]))
                logger.info("Episode {} termination reason: {}".format(ts, failed_reason))
                logger.info("Episode {} length: {}".format(ts, step))
                logger.info("Episode {} Decision Step: {}".format(ts, local_decision_step))
                logger.info("Episode {} Success Decision: {}".format(ts, local_succ_decision))
                logger.info("Episode {} Policy Action Hist: {}".format(ts, local_policy_action_hist))
                logger.info("Episode {} Expert Action Hist: {}".format(ts, local_expert_action_hist))
                logger.info(
                    "Episode {} Stop Counts policy/expert/matched: {}/{}/{}".format(
                        ts, local_predicted_stop, local_expert_stop, local_matched_stop
                    )
                )


            mean_reward = np.mean(rewards_list)
            sr = np.mean(success)
            mSuccD, aSuccD, SuccDAct = cal_step_metric(decision_step, succ_decision)
            logger.info(f"Step: {self.n_calls} - Success Rate: {sr} - Mean Reward: {mean_reward} \n")
            logger.info("Mean Decision Succ: {}".format(mSuccD))
            logger.info("Average Decision Succ: {}".format(aSuccD))
            logger.info("Decision Succ for each action: {}".format(SuccDAct))
            logger.info("Mean Episode Length: {}".format(float(np.mean(episode_lengths)) if episode_lengths else 0.0))
            logger.info("Termination Counts fail/overtime/truncated: {}/{}/{}".format(fail_episodes, timeout_episodes, truncated_episodes))
            logger.info("Policy Action Hist: {}".format(policy_action_hist))
            logger.info("Expert Action Hist: {}".format(expert_action_hist))
            logger.info(
                "Stop Counts policy/expert/matched: {}/{}/{}".format(
                    predicted_stop_total, expert_stop_total, matched_stop_total
                )
            )
            if episode_lengths and oracle_steps:
                logger.info(
                    "Mean PPO Steps / Mean Oracle Steps: {}/{}".format(
                        float(np.mean(episode_lengths)),
                        float(np.mean(oracle_steps)),
                    )
                )

            # Log the mean reward
            with open(os.path.join(self.save_path, "{}_eval_rewards.txt".format(self.exp_name)), "a") as file:
                file.write(f"Step: {self.n_calls} - Success Rate: {sr} - Mean Reward: {mean_reward} \n")
                file.write("Mean Decision Succ: {}\n".format(mSuccD))
                file.write("Average Decision Succ: {}\n".format(aSuccD))
                file.write("Decision Succ for each action: {}\n".format(SuccDAct))
                file.write("Mean Episode Length: {}\n".format(float(np.mean(episode_lengths)) if episode_lengths else 0.0))
                file.write("Termination Counts fail/overtime/truncated: {}/{}/{}\n".format(fail_episodes, timeout_episodes, truncated_episodes))
                file.write("Policy Action Hist: {}\n".format(policy_action_hist))
                file.write("Expert Action Hist: {}\n".format(expert_action_hist))
                file.write("Stop Counts policy/expert/matched: {}/{}/{}\n".format(predicted_stop_total, expert_stop_total, matched_stop_total))
                if episode_lengths and oracle_steps:
                    file.write(
                        "Mean PPO Steps / Mean Oracle Steps: {}/{}\n".format(
                            float(np.mean(episode_lengths)),
                            float(np.mean(oracle_steps)),
                        )
                    )
            extra_metrics = {
                "fail": fail_episodes,
                "timeout": timeout_episodes,
                "truncated": truncated_episodes,
                "policy_stop_count": predicted_stop_total,
                "expert_stop_count": expert_stop_total,
                "matched_stop_count": matched_stop_total,
                "policy_action_hist_0": policy_action_hist.get(0, 0),
                "policy_action_hist_1": policy_action_hist.get(1, 0),
                "policy_action_hist_2": policy_action_hist.get(2, 0),
                "policy_action_hist_3": policy_action_hist.get(3, 0),
                "expert_action_hist_0": expert_action_hist.get(0, 0),
                "expert_action_hist_1": expert_action_hist.get(1, 0),
                "expert_action_hist_2": expert_action_hist.get(2, 0),
                "expert_action_hist_3": expert_action_hist.get(3, 0),
                "shield_intervention_count": 0,
                "shield_intervention_rate": 0.0,
                "elapsed_wall_clock_seconds": float(time.time() - self.start_time),
                "seconds_per_1k_timesteps": (
                    float((time.time() - self.start_time) / max(self.num_timesteps, 1) * 1000.0)
                ),
                "train_fail": int(self.train_fail),
                "train_timeout": int(self.train_timeout),
                "train_success": int(self.train_success),
                "train_truncated": int(self.train_truncated),
                "train_completed": int(self.train_completed),
                "train_fail_since_last_eval": int(self.train_fail - self._last_eval_train_fail),
                "train_timeout_since_last_eval": int(self.train_timeout - self._last_eval_train_timeout),
                "train_success_since_last_eval": int(self.train_success - self._last_eval_train_success),
                "train_truncated_since_last_eval": int(self.train_truncated - self._last_eval_train_truncated),
                "train_completed_since_last_eval": int(self.train_completed - self._last_eval_train_completed),
            }
            if hasattr(self.model, "get_shield_metrics"):
                extra_metrics.update(self.model.get_shield_metrics())
            self._append_metrics_csv(
                self.n_calls,
                sr,
                SuccDAct,
                extra_metrics=extra_metrics,
                mean_reward=mean_reward,
            )
            self._last_eval_train_fail = self.train_fail
            self._last_eval_train_timeout = self.train_timeout
            self._last_eval_train_success = self.train_success
            self._last_eval_train_truncated = self.train_truncated
            self._last_eval_train_completed = self.train_completed

            # Update the best model if current mean reward is better
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save("{}/best_model.zip".format(self.save_path))

        return True

class DreamerEvalCheckpointCallback(CheckpointCallback):
    def __init__(self, exp_name, simulation_config, episode_data, eval_actions, eval_freq=50000, *args, **kwargs):
        super(DreamerEvalCheckpointCallback, self).__init__(*args, **kwargs)
        self.eval_freq = eval_freq
        self.exp_name = exp_name
        self.best_mean_reward = -np.inf
        self.start_time = time.time()
        self.simulation_config = simulation_config
        with open(episode_data, "rb") as f:
            self.episode_data = pkl.load(f)
        self.eval_actions = eval_actions
        self.stop_action_id = int(self.eval_actions.get("Stop", next(iter(self.eval_actions.values()))))
        self.csv_path = os.path.join(self.save_path, f"{self.exp_name}_metrics.csv")

    def _append_metrics_csv(self, step, sr, SuccDAct, mean_reward=None):
        os.makedirs(self.save_path, exist_ok=True)
        fieldnames = [
            "step",
            "tsr",
            "dsr",
            "mean_reward",
            "elapsed_wall_clock_seconds",
            "seconds_per_1k_timesteps",
        ]
        row = {
            "step": step,
            "tsr": sr,
            "dsr": SuccDAct.get(self.stop_action_id, 0.0),
            "mean_reward": mean_reward,
            "elapsed_wall_clock_seconds": float(time.time() - self.start_time),
            "seconds_per_1k_timesteps": float((time.time() - self.start_time) / max(self.num_timesteps, 1) * 1000.0),
        }
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
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
        if self.n_calls % self.eval_freq == 0:
            rewards_list = []
            success = []
            decision_step = {}
            succ_decision = {}
            for action, id in self.eval_actions.items():
                decision_step[id] = 0
                succ_decision[id] = 0
            for ts in list(self.episode_data.keys()):  # Number of episodes for evaluation
                logger.info("Evaluating episode {}...".format(ts))
                episode_cache = self.episode_data[ts]
                if "label_info" in episode_cache:
                    logger.info("Episode label: {}".format(episode_cache["label_info"]))
                max_steps = episode_cache["label_info"]["oracle_step"] * 2
                eval_env = make_env(self.simulation_config, episode_cache, False)
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
                    if oracle_action in local_decision_step.keys():
                        local_decision_step[oracle_action] = 1
                        if int(env_action) != oracle_action:
                            local_succ_decision[oracle_action] = 0
                    obs, reward, done, info = eval_env.step(int(env_action))
                    if info["Fail"][0]:
                        episode_rewards += reward
                        break
                    episode_rewards += reward
                    step += 1
                if info["success"]:
                    logger.info("Episode {} success.".format(ts))
                    success.append(1)
                else:
                    logger.info("Episode {} failed.".format(ts))
                    success.append(0)
                if step >= max_steps:
                    episode_rewards -= 3
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
            mSuccD, aSuccD, SuccDAct = cal_step_metric(decision_step, succ_decision)
            logger.info(f"Step: {self.n_calls} - Success Rate: {sr} - Mean Reward: {mean_reward} \n")
            logger.info("Mean Decision Succ: {}".format(mSuccD))
            logger.info("Average Decision Succ: {}".format(aSuccD))
            logger.info("Decision Succ for each action: {}".format(SuccDAct))

            # Log the mean reward
            with open(os.path.join(self.save_path, "{}_eval_rewards.txt".format(self.exp_name)), "a") as file:
                file.write(f"Step: {self.n_calls} - Success Rate: {sr} - Mean Reward: {mean_reward} \n")
                file.write("Mean Decision Succ: {}\n".format(mSuccD))
                file.write("Average Decision Succ: {}\n".format(aSuccD))
                file.write("Decision Succ for each action: {}\n".format(SuccDAct))
            self._append_metrics_csv(self.n_calls, sr, SuccDAct, mean_reward=mean_reward)

            # Update the best model if current mean reward is better
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save("{}/best_model.zip".format(self.save_path))

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
