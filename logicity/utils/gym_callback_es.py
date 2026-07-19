from stable_baselines3.common.callbacks import CheckpointCallback
from logicity.utils.load import CityLoader
from logicity.utils.gym_wrapper_es import GymCityWrapperES
import numpy as np
import torch
import os
import logging
import pickle as pkl
logger = logging.getLogger(__name__)


def _unpack_step_result(step_result):
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return obs, reward, (terminated or truncated), info
    return step_result

def make_env(simulation_config, episode_cache=None, return_cache=False): 
    # Unpack arguments from simulation_config and pass them to CityLoader
    city, cached_observation = CityLoader.from_yaml(**simulation_config, episode_cache=episode_cache)
    env = GymCityWrapperES(city)
    if return_cache: 
        return env, cached_observation
    else:
        return env

class EvalCheckpointCallbackES(CheckpointCallback):
    def __init__(self, exp_name, simulation_config, episode_data, eval_actions, eval_freq=50000, *args, **kwargs):
        super(EvalCheckpointCallbackES, self).__init__(*args, **kwargs)
        self.eval_freq = eval_freq
        self.exp_name = exp_name
        self.best_mean_reward = -np.inf
        self.simulation_config = simulation_config
        with open(episode_data, "rb") as f:
            self.episode_data = pkl.load(f)
        self.eval_actions = eval_actions

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
                local_decision_step = {}
                local_succ_decision = {}
                for acc, id in self.eval_actions.items():
                    local_decision_step[id] = 0
                    local_succ_decision[id] = 1
                done = False
                while (not done) and (step < max_steps):
                    oracle_action = eval_env.expert_action
                    action, _states = self.model.predict(obs, deterministic=True)
                    if oracle_action in local_decision_step.keys():
                        local_decision_step[oracle_action] = 1
                        if int(action) != oracle_action:
                            local_succ_decision[oracle_action] = 0
                    obs, reward, done, info = _unpack_step_result(eval_env.step(int(action)))
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
                    obs, reward, done, info = _unpack_step_result(eval_env.step(int(env_action)))
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
