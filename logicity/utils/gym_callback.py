from stable_baselines3.common.callbacks import CheckpointCallback
from logicity.utils.load import CityLoader
from logicity.utils.gym_wrapper import GymCityWrapper
import numpy as np
import torch
import os
import logging
import pickle as pkl
import csv
import yaml
logger = logging.getLogger(__name__)

def _sanitize_rule_name(rule_name):
    return str(rule_name).replace(" ", "_")

def _load_task_rule_names(rule_yaml_file):
    with open(rule_yaml_file, "r") as handle:
        data = yaml.safe_load(handle) or {}
    return [rule["name"] for rule in data.get("Rules", {}).get("Task", []) if "name" in rule]

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
        self.simulation_config = simulation_config
        with open(episode_data, "rb") as f:
            self.episode_data = pkl.load(f)
        self.eval_actions = eval_actions
        self.eval_csv_path = os.path.join(self.save_path, "{}_eval_metrics.csv".format(self.exp_name))
        self._last_eval_timestep = 0
        self._last_save_timestep = 0
        self.failure_rule_names = _load_task_rule_names(self.simulation_config["rule_yaml_file"])

    def _append_eval_csv_row(self, row_dict):
        file_exists = os.path.isfile(self.eval_csv_path)
        base_fieldnames = [
            "timestep",
            "num_eval_episodes",
            "tsr",
            "mean_reward",
            "failure_rate",
            "timeout_rate",
            "mean_episode_length",
            "action_0_count",
            "action_1_count",
            "action_2_count",
            "action_3_count",
            "mean_decision_succ",
        ]
        decision_fields = sorted([k for k in row_dict.keys() if k.startswith("decision_succ_action_")])
        fail_rule_fields = ["fail_rule_{}".format(_sanitize_rule_name(rule_name)) for rule_name in self.failure_rule_names]
        fieldnames = base_fieldnames + decision_fields + fail_rule_fields
        with open(self.eval_csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_dict)

    def _on_step(self) -> bool:
        learning_starts = getattr(self.model, "learning_starts", 0)
        current_timestep = self.model.num_timesteps
        if (current_timestep - self._last_save_timestep) >= self.save_freq and (current_timestep > learning_starts):
            self._last_save_timestep = current_timestep
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
        if (current_timestep - self._last_eval_timestep) >= self.eval_freq:
            self._last_eval_timestep = current_timestep
            rewards_list = []
            success = []
            failures = []
            timeouts = []
            episode_lengths = []
            decision_step = {}
            succ_decision = {}
            action_hist = {0: 0, 1: 0, 2: 0, 3: 0}
            fail_rule_counts = {}
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
                local_decision_step = {}
                local_succ_decision = {}
                for acc, id in self.eval_actions.items():
                    local_decision_step[id] = 0
                    local_succ_decision[id] = 1
                done = False
                while (not done) and (step < max_steps):
                    oracle_action = eval_env.expert_action
                    action, _states = self.model.predict(obs, deterministic=True)
                    action_hist[int(action)] = action_hist.get(int(action), 0) + 1
                    if oracle_action in local_decision_step.keys():
                        local_decision_step[oracle_action] = 1
                        if int(action) != oracle_action:
                            local_succ_decision[oracle_action] = 0
                    obs, reward, done, info = eval_env.step(int(action))
                    if info["Fail"][0]:
                        episode_rewards += reward
                        break
                    episode_rewards += reward
                    step += 1
                if info["is_success"]:
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
            failure_rate = np.mean(failures)
            timeout_rate = np.mean(timeouts)
            mean_length = np.mean(episode_lengths)
            mSuccD, aSuccD, SuccDAct = cal_step_metric(decision_step, succ_decision)
            logger.info(f"Timestep: {current_timestep} - Success Rate: {sr} - Mean Reward: {mean_reward} \n")
            logger.info("Failure Rate: {} - Timeout Rate: {} - Mean Episode Length: {}".format(failure_rate, timeout_rate, mean_length))
            logger.info("Mean Decision Succ: {}".format(mSuccD))
            logger.info("Average Decision Succ: {}".format(aSuccD))
            logger.info("Decision Succ for each action: {}".format(SuccDAct))
            logger.info("Action Histogram: {}".format(action_hist))
            logger.info("Failure Rule Counts: {}".format(fail_rule_counts))

            # Log the mean reward
            with open(os.path.join(self.save_path, "{}_eval_rewards.txt".format(self.exp_name)), "a") as file:
                file.write(f"Timestep: {current_timestep} - Success Rate: {sr} - Mean Reward: {mean_reward} \n")
                file.write("Failure Rate: {} - Timeout Rate: {} - Mean Episode Length: {}\n".format(failure_rate, timeout_rate, mean_length))
                file.write("Mean Decision Succ: {}\n".format(mSuccD))
                file.write("Average Decision Succ: {}\n".format(aSuccD))
                file.write("Decision Succ for each action: {}\n".format(SuccDAct))
                file.write("Action Histogram: {}\n".format(action_hist))
                file.write("Failure Rule Counts: {}\n".format(fail_rule_counts))

            csv_row = {
                "timestep": current_timestep,
                "num_eval_episodes": len(rewards_list),
                "tsr": sr,
                "mean_reward": mean_reward,
                "failure_rate": failure_rate,
                "timeout_rate": timeout_rate,
                "mean_episode_length": mean_length,
                "action_0_count": action_hist.get(0, 0),
                "action_1_count": action_hist.get(1, 0),
                "action_2_count": action_hist.get(2, 0),
                "action_3_count": action_hist.get(3, 0),
                "mean_decision_succ": mSuccD,
            }
            for action_id, value in SuccDAct.items():
                csv_row["decision_succ_action_{}".format(action_id)] = value
            for rule_name in self.failure_rule_names:
                csv_row["fail_rule_{}".format(_sanitize_rule_name(rule_name))] = fail_rule_counts.get(rule_name, 0)
            self._append_eval_csv_row(csv_row)

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
        self.simulation_config = simulation_config
        with open(episode_data, "rb") as f:
            self.episode_data = pkl.load(f)
        self.eval_actions = eval_actions
        self.eval_csv_path = os.path.join(self.save_path, "{}_eval_metrics.csv".format(self.exp_name))
        self._last_eval_timestep = 0
        self._last_save_timestep = 0
        self.failure_rule_names = _load_task_rule_names(self.simulation_config["rule_yaml_file"])

    def _append_eval_csv_row(self, row_dict):
        file_exists = os.path.isfile(self.eval_csv_path)
        base_fieldnames = [
            "timestep",
            "num_eval_episodes",
            "tsr",
            "mean_reward",
            "failure_rate",
            "timeout_rate",
            "mean_episode_length",
            "action_0_count",
            "action_1_count",
            "action_2_count",
            "action_3_count",
            "mean_decision_succ",
        ]
        decision_fields = sorted([k for k in row_dict.keys() if k.startswith("decision_succ_action_")])
        fail_rule_fields = ["fail_rule_{}".format(_sanitize_rule_name(rule_name)) for rule_name in self.failure_rule_names]
        fieldnames = base_fieldnames + decision_fields + fail_rule_fields
        with open(self.eval_csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_dict)

    def _on_step(self) -> bool:
        current_timestep = self.model.num_timesteps
        if (current_timestep - self._last_save_timestep) >= self.save_freq:
            self._last_save_timestep = current_timestep
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
        if (current_timestep - self._last_eval_timestep) >= self.eval_freq:
            self._last_eval_timestep = current_timestep
            rewards_list = []
            success = []
            failures = []
            timeouts = []
            episode_lengths = []
            decision_step = {}
            succ_decision = {}
            action_hist = {0: 0, 1: 0, 2: 0, 3: 0}
            fail_rule_counts = {}
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
                    action_hist[int(env_action)] = action_hist.get(int(env_action), 0) + 1
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
            failure_rate = np.mean(failures)
            timeout_rate = np.mean(timeouts)
            mean_length = np.mean(episode_lengths)
            mSuccD, aSuccD, SuccDAct = cal_step_metric(decision_step, succ_decision)
            logger.info(f"Timestep: {current_timestep} - Success Rate: {sr} - Mean Reward: {mean_reward} \n")
            logger.info("Failure Rate: {} - Timeout Rate: {} - Mean Episode Length: {}".format(failure_rate, timeout_rate, mean_length))
            logger.info("Mean Decision Succ: {}".format(mSuccD))
            logger.info("Average Decision Succ: {}".format(aSuccD))
            logger.info("Decision Succ for each action: {}".format(SuccDAct))
            logger.info("Action Histogram: {}".format(action_hist))
            logger.info("Failure Rule Counts: {}".format(fail_rule_counts))

            # Log the mean reward
            with open(os.path.join(self.save_path, "{}_eval_rewards.txt".format(self.exp_name)), "a") as file:
                file.write(f"Timestep: {current_timestep} - Success Rate: {sr} - Mean Reward: {mean_reward} \n")
                file.write("Failure Rate: {} - Timeout Rate: {} - Mean Episode Length: {}\n".format(failure_rate, timeout_rate, mean_length))
                file.write("Mean Decision Succ: {}\n".format(mSuccD))
                file.write("Average Decision Succ: {}\n".format(aSuccD))
                file.write("Decision Succ for each action: {}\n".format(SuccDAct))
                file.write("Action Histogram: {}\n".format(action_hist))
                file.write("Failure Rule Counts: {}\n".format(fail_rule_counts))

            csv_row = {
                "timestep": current_timestep,
                "num_eval_episodes": len(rewards_list),
                "tsr": sr,
                "mean_reward": mean_reward,
                "failure_rate": failure_rate,
                "timeout_rate": timeout_rate,
                "mean_episode_length": mean_length,
                "action_0_count": action_hist.get(0, 0),
                "action_1_count": action_hist.get(1, 0),
                "action_2_count": action_hist.get(2, 0),
                "action_3_count": action_hist.get(3, 0),
                "mean_decision_succ": mSuccD,
            }
            for action_id, value in SuccDAct.items():
                csv_row["decision_succ_action_{}".format(action_id)] = value
            for rule_name in self.failure_rule_names:
                csv_row["fail_rule_{}".format(_sanitize_rule_name(rule_name))] = fail_rule_counts.get(rule_name, 0)
            self._append_eval_csv_row(csv_row)

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
