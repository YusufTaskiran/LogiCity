import warnings
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

from logicity.plpg import ProbLogCircuitShield


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
        self._plpg_shield = None
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
        else:
            pred_grounding_index = self.env.pred_grounding_index
        self._plpg_shield = ProbLogCircuitShield(
            self.plpg_template,
            pred_grounding_index,
            rule_yaml_file=self.plpg_rule_yaml_file,
            sensor_noise=self.plpg_sensor_noise,
        )

    def _base_and_shielded(self, obs_tensor: th.Tensor):
        base_dist = self.policy.get_distribution(obs_tensor)
        base_probs = base_dist.distribution.probs
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
        intervened = (l1_shift > 1e-8).float()
        hazard = shield_info.get("template_metrics", {}).get(
            "hazard",
            th.zeros(base_probs.shape[0], dtype=base_probs.dtype, device=base_probs.device),
        )
        forced_stop = ((shielded_action == 3) & (hazard > 0.5)).float()

        return {
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

        n_steps = 0
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                _, shield_info, shielded_dist, values = self._base_and_shielded(obs_tensor)
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
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses, safety_losses = [], [], []
        clip_fractions = []
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                _, shield_info, shielded_dist, values = self._base_and_shielded(rollout_data.observations)
                values = values.flatten()
                log_prob = shielded_dist.log_prob(actions)
                entropy = shielded_dist.entropy()

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
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        self.policy.set_training_mode(False)
        obs_tensor, vectorized_env = self.policy.obs_to_tensor(observation)
        with th.no_grad():
            _, _, shielded_dist, _ = self._base_and_shielded(obs_tensor)
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
            base_probs, shield_info, shielded_dist, _ = self._base_and_shielded(obs_tensor)
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
