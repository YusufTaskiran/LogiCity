import warnings
import copy
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.distributions import Categorical
from torch.nn import functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor

from logicity.plpg import CentralizedJointTrafficShield, ProbLogCircuitShield


class PLPGPPO(PPO):
    """
    PLPG implementation for the current easy-lite SPF setting.

    This first version uses:
    - base PPO actor-critic network
    - easy-task binary safety abstraction
    - shielded action distribution pi+
    - safety regularization term
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        plpg_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.plpg_kwargs = plpg_kwargs or {}
        self.plpg_alpha = self.plpg_kwargs.get("alpha", 0.1)
        self.safety_loss_source = self.plpg_kwargs.get("safety_loss_source", "shielded")
        self.plpg_template = self.plpg_kwargs.get("template", "easy")
        self.plpg_rule_yaml_file = self.plpg_kwargs.get("rule_yaml_file")
        self.plpg_sensor_noise = self.plpg_kwargs.get("sensor_noise")
        self.plpg_shield_scope = self.plpg_kwargs.get("shield_scope", "decentralized")
        self._plpg_shield = None
        self._plpg_joint_context = None
        self._plpg_rollout_joint_contexts = []
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        self._setup_plpg()

    def _setup_plpg(self):
        if hasattr(self.env, "get_attr"):
            pred_grounding_index = self.env.get_attr("pred_grounding_index")[0]
            num_agents = self.env.num_envs
        else:
            pred_grounding_index = self.env.pred_grounding_index
            num_agents = 1
        if self.plpg_shield_scope == "centralized":
            self._plpg_shield = CentralizedJointTrafficShield(
                self.plpg_template,
                pred_grounding_index,
                rule_yaml_file=self.plpg_rule_yaml_file,
                sensor_noise=self.plpg_sensor_noise,
                num_agents=num_agents,
            )
        else:
            self._plpg_shield = ProbLogCircuitShield(
                self.plpg_template,
                pred_grounding_index,
                rule_yaml_file=self.plpg_rule_yaml_file,
                sensor_noise=self.plpg_sensor_noise,
            )

    def set_plpg_joint_context(self, joint_context):
        self._plpg_joint_context = copy.deepcopy(joint_context)

    def _get_joint_context_from_env(self, env):
        if self.plpg_shield_scope != "centralized":
            return None
        if hasattr(env, "get_attr"):
            contexts = env.get_attr("last_joint_shield_context")
            if contexts:
                return copy.deepcopy(contexts[0])
        elif hasattr(env, "last_joint_shield_context"):
            return copy.deepcopy(env.last_joint_shield_context)
        return copy.deepcopy(self._plpg_joint_context)

    def _base_and_shielded(self, obs_tensor: th.Tensor, joint_context=None):
        base_dist = self.policy.get_distribution(obs_tensor)
        base_probs = base_dist.distribution.probs
        if self.plpg_shield_scope == "centralized":
            shield_info = self._plpg_shield.shield_policy(base_probs, obs_tensor, joint_context=joint_context)
        else:
            shield_info = self._plpg_shield.shield_policy(base_probs, obs_tensor)
        shielded_probs = shield_info["shielded_probs"]
        shielded_dist = Categorical(probs=shielded_probs)
        values = self.policy.predict_values(obs_tensor)
        return base_probs, shield_info, shielded_dist, values

    def _extract_plpg_step_info(self, base_probs, shield_info, deterministic=False):
        shielded_probs = shield_info["shielded_probs"]
        if deterministic:
            base_action = th.argmax(base_probs, dim=1)
            shielded_action = th.argmax(shielded_probs, dim=1)
        else:
            base_action = Categorical(probs=base_probs).sample()
            shielded_action = Categorical(probs=shielded_probs).sample()

        kl_base_to_shielded = th.sum(
            base_probs * (th.log(base_probs.clamp_min(1e-8)) - th.log(shielded_probs.clamp_min(1e-8))),
            dim=1,
        )
        l1_shift = th.sum(th.abs(base_probs - shielded_probs), dim=1)
        # Ignore tiny floating-point renormalization noise when counting interventions.
        intervened = (l1_shift > 1e-4).float()
        hazard = shield_info.get("template_metrics", {}).get(
            "hazard",
            th.zeros(base_probs.shape[0], dtype=base_probs.dtype, device=base_probs.device),
        )
        forced_stop = ((shielded_action == 3) & (hazard > 0.5)).float()

        step_info = {
            "base_action": base_action,
            "shielded_action": shielded_action,
            "base_probs": base_probs,
            "shielded_probs": shielded_probs,
            "safety_probs": shield_info["safety_probs"],
            "base_policy_safe_prob": shield_info["base_policy_safe_prob"],
            "shielded_policy_safe_prob": shield_info["shielded_policy_safe_prob"],
            "kl_base_to_shielded": kl_base_to_shielded,
            "l1_shift_base_to_shielded": l1_shift,
            "intervened": intervened,
            "hazard": hazard,
            "forced_stop": forced_stop,
        }
        if "joint_debug" in shield_info:
            step_info["joint_debug"] = shield_info["joint_debug"]
        joint_hazard = shield_info.get("template_metrics", {}).get("joint_hazard")
        if joint_hazard is not None:
            step_info["joint_hazard"] = joint_hazard
        return step_info

    def collect_rollouts(
        self,
        env,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        rollout_buffer.reset()

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        if self.plpg_shield_scope == "centralized":
            self._plpg_rollout_joint_contexts = []

        n_steps = 0
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                joint_context = self._get_joint_context_from_env(env)
                if self.plpg_shield_scope == "centralized":
                    self._plpg_rollout_joint_contexts.append(copy.deepcopy(joint_context))
                _, shield_info, shielded_dist, values = self._base_and_shielded(obs_tensor, joint_context=joint_context)
                actions = shielded_dist.sample()
                log_probs = shielded_dist.log_prob(actions)

            actions_np = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions_np)

            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions_buffer = actions.reshape(-1, 1)
            else:
                actions_buffer = actions

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions_buffer,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        if self.plpg_shield_scope == "centralized":
            return self._train_centralized()
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses, safety_losses = [], [], []
        clip_fractions = []
        mean_base_probs = []
        mean_shielded_probs = []
        mean_prob_shifts = []
        mean_total_variation_shifts = []
        mean_stop_bias_shifts = []
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                base_probs, shield_info, shielded_dist, values = self._base_and_shielded(rollout_data.observations)
                values = values.flatten()
                log_prob = shielded_dist.log_prob(actions)
                entropy = shielded_dist.entropy()
                shielded_probs = shielded_dist.probs

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )

                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                if self.safety_loss_source == "base":
                    safe_prob = shield_info["base_policy_safe_prob"]
                else:
                    safe_prob = shield_info["shielded_policy_safe_prob"]
                safety_loss = -th.log(safe_prob).mean()
                safety_losses.append(safety_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.plpg_alpha * safety_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                    mean_base_probs.append(base_probs.mean(dim=0).detach().cpu().numpy())
                    mean_shielded_probs.append(shielded_probs.mean(dim=0).detach().cpu().numpy())
                    mean_prob_shifts.append((shielded_probs - base_probs).mean(dim=0).detach().cpu().numpy())
                    mean_total_variation_shifts.append(
                        0.5 * th.abs(shielded_probs - base_probs).sum(dim=1).mean().item()
                    )
                    mean_stop_bias_shifts.append(
                        (shielded_probs[:, 3] - base_probs[:, 3]).mean().item()
                    )

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/safety_loss", np.mean(safety_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        self.logger.record("train/plpg_alpha", self.plpg_alpha)
        if len(mean_base_probs) > 0:
            avg_base_probs = np.mean(np.asarray(mean_base_probs), axis=0)
            avg_shielded_probs = np.mean(np.asarray(mean_shielded_probs), axis=0)
            avg_prob_shifts = np.mean(np.asarray(mean_prob_shifts), axis=0)
            action_names = getattr(self._plpg_shield, "action_names", [str(i) for i in range(len(avg_base_probs))])
            for action_id, action_name in enumerate(action_names):
                self.logger.record(f"train/base_prob_{action_name}", float(avg_base_probs[action_id]))
                self.logger.record(f"train/shielded_prob_{action_name}", float(avg_shielded_probs[action_id]))
                self.logger.record(f"train/prob_shift_{action_name}", float(avg_prob_shifts[action_id]))
        if len(mean_total_variation_shifts) > 0:
            self.logger.record("train/mean_total_variation_shift", float(np.mean(mean_total_variation_shifts)))
            self.logger.record("train/mean_stop_bias_shift", float(np.mean(mean_stop_bias_shifts)))
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def _train_centralized(self) -> None:
        if len(self._plpg_rollout_joint_contexts) != self.rollout_buffer.buffer_size:
            raise ValueError(
                "Centralized PLPG rollout contexts ({}) do not match rollout buffer size ({}).".format(
                    len(self._plpg_rollout_joint_contexts),
                    self.rollout_buffer.buffer_size,
                )
            )

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses, safety_losses = [], [], []
        clip_fractions = []
        mean_base_probs = []
        mean_shielded_probs = []
        mean_prob_shifts = []
        mean_total_variation_shifts = []
        mean_stop_bias_shifts = []
        continue_training = True
        approx_kl_divs = []

        observations = th.as_tensor(self.rollout_buffer.observations, device=self.device)
        actions = th.as_tensor(self.rollout_buffer.actions, device=self.device).long().squeeze(-1)
        old_log_prob = th.as_tensor(self.rollout_buffer.log_probs, device=self.device)
        returns = th.as_tensor(self.rollout_buffer.returns, device=self.device)
        old_values = th.as_tensor(self.rollout_buffer.values, device=self.device)
        advantages = th.as_tensor(self.rollout_buffer.advantages, device=self.device)

        n_steps, num_agents = observations.shape[:2]
        flat_obs = observations.reshape(n_steps * num_agents, *observations.shape[2:])

        for epoch in range(self.n_epochs):
            if self.use_sde:
                self.policy.reset_noise(n_steps * num_agents)

            base_dist = self.policy.get_distribution(flat_obs)
            base_probs = base_dist.distribution.probs.reshape(n_steps, num_agents, -1)
            values = self.policy.predict_values(flat_obs).reshape(n_steps, num_agents)

            shielded_probs_per_step = []
            base_safe_per_step = []
            shielded_safe_per_step = []
            for step_idx in range(n_steps):
                shield_info = self._plpg_shield.shield_policy(
                    base_probs[step_idx],
                    observations[step_idx],
                    joint_context=self._plpg_rollout_joint_contexts[step_idx],
                )
                shielded_probs_per_step.append(shield_info["shielded_probs"])
                base_safe_per_step.append(shield_info["base_policy_safe_prob"])
                shielded_safe_per_step.append(shield_info["shielded_policy_safe_prob"])

            shielded_probs = th.stack(shielded_probs_per_step, dim=0)
            base_safe_prob = th.stack(base_safe_per_step, dim=0)
            shielded_safe_prob = th.stack(shielded_safe_per_step, dim=0)
            shielded_dist = Categorical(probs=shielded_probs.reshape(n_steps * num_agents, -1))
            log_prob = shielded_dist.log_prob(actions.reshape(-1)).reshape(n_steps, num_agents)
            entropy = shielded_dist.entropy().reshape(n_steps, num_agents)

            norm_advantages = advantages
            if self.normalize_advantage and norm_advantages.numel() > 1:
                norm_advantages = (norm_advantages - norm_advantages.mean()) / (norm_advantages.std() + 1e-8)

            ratio = th.exp(log_prob - old_log_prob)
            policy_loss_1 = norm_advantages * ratio
            policy_loss_2 = norm_advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

            pg_losses.append(policy_loss.item())
            clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
            clip_fractions.append(clip_fraction)

            if self.clip_range_vf is None:
                values_pred = values
            else:
                values_pred = old_values + th.clamp(values - old_values, -clip_range_vf, clip_range_vf)

            value_loss = F.mse_loss(returns, values_pred)
            value_losses.append(value_loss.item())

            entropy_loss = -th.mean(entropy)
            entropy_losses.append(entropy_loss.item())

            if self.safety_loss_source == "base":
                safe_prob = base_safe_prob
            else:
                safe_prob = shielded_safe_prob
            safety_loss = -th.log(safe_prob.clamp_min(1e-8)).mean()
            safety_losses.append(safety_loss.item())

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.plpg_alpha * safety_loss

            with th.no_grad():
                log_ratio = log_prob - old_log_prob
                approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                approx_kl_divs.append(approx_kl_div)
                mean_base_probs.append(base_probs.mean(dim=(0, 1)).detach().cpu().numpy())
                mean_shielded_probs.append(shielded_probs.mean(dim=(0, 1)).detach().cpu().numpy())
                mean_prob_shifts.append((shielded_probs - base_probs).mean(dim=(0, 1)).detach().cpu().numpy())
                mean_total_variation_shifts.append(
                    0.5 * th.abs(shielded_probs - base_probs).sum(dim=2).mean().item()
                )
                mean_stop_bias_shifts.append((shielded_probs[:, :, 3] - base_probs[:, :, 3]).mean().item())

            if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                continue_training = False
                if self.verbose >= 1:
                    print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                break

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            self._n_updates += 1

            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/safety_loss", np.mean(safety_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        self.logger.record("train/plpg_alpha", self.plpg_alpha)
        if len(mean_base_probs) > 0:
            avg_base_probs = np.mean(np.asarray(mean_base_probs), axis=0)
            avg_shielded_probs = np.mean(np.asarray(mean_shielded_probs), axis=0)
            avg_prob_shifts = np.mean(np.asarray(mean_prob_shifts), axis=0)
            action_names = getattr(self._plpg_shield, "action_names", [str(i) for i in range(len(avg_base_probs))])
            for action_id, action_name in enumerate(action_names):
                self.logger.record(f"train/base_prob_{action_name}", float(avg_base_probs[action_id]))
                self.logger.record(f"train/shielded_prob_{action_name}", float(avg_shielded_probs[action_id]))
                self.logger.record(f"train/prob_shift_{action_name}", float(avg_prob_shifts[action_id]))
        if len(mean_total_variation_shifts) > 0:
            self.logger.record("train/mean_total_variation_shift", float(np.mean(mean_total_variation_shifts)))
            self.logger.record("train/mean_stop_bias_shift", float(np.mean(mean_stop_bias_shifts)))
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        self.policy.set_training_mode(False)
        obs_tensor, vectorized_env = self.policy.obs_to_tensor(observation)
        with th.no_grad():
            _, _, shielded_dist, _ = self._base_and_shielded(obs_tensor, joint_context=self._plpg_joint_context)
            if deterministic:
                actions = th.argmax(shielded_dist.probs, dim=1)
            else:
                actions = shielded_dist.sample()
        actions = actions.cpu().numpy()
        if not vectorized_env:
            actions = actions.squeeze(axis=0)
        return actions, state

    def predict_with_plpg_info(self, observation, state=None, episode_start=None, deterministic=False):
        self.policy.set_training_mode(False)
        obs_tensor, vectorized_env = self.policy.obs_to_tensor(observation)
        with th.no_grad():
            base_probs, shield_info, shielded_dist, _ = self._base_and_shielded(
                obs_tensor,
                joint_context=self._plpg_joint_context,
            )
            step_info = self._extract_plpg_step_info(base_probs, shield_info, deterministic=deterministic)
            if deterministic:
                actions = step_info["shielded_action"]
            else:
                actions = shielded_dist.sample()
        actions_np = actions.cpu().numpy()
        if not vectorized_env:
            actions_np = actions_np.squeeze(axis=0)
        info_np = {}
        for key, value in step_info.items():
            if isinstance(value, th.Tensor):
                value = value.detach().cpu().numpy()
                if not vectorized_env and value.shape[0] == 1:
                    value = value.squeeze(axis=0)
            info_np[key] = value
        return actions_np, state, info_np
