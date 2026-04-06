import copy
from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.utils import obs_as_tensor

from ...shields import ProbabilisticLogicShield


def _unwrap_first_env(env: Any) -> Any:
    current = env
    if hasattr(current, "envs") and len(getattr(current, "envs")) > 0:
        current = current.envs[0]
    while hasattr(current, "env") and not hasattr(current, "shield_config"):
        current = current.env
    return current


class PLSPPO(SB3PPO):
    """Probabilistic Logic Shield PPO.

    Uses the same uncertain shield-side sensor model as DLS, but computes
    a soft conflict probability and renormalizes the policy with soft action
    safety values instead of a hard deterministic mask.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pls_enabled = False
        self._pls_pred_grounding_index = None
        self._shield_cfg = None
        self._shield = None
        self._safety_coefficient = 0.1
        self._init_pls_from_env()

    def _init_pls_from_env(self) -> None:
        if self.env is None:
            return
        try:
            base_env = _unwrap_first_env(self.env)
            shield_cfg = getattr(base_env, "shield_config", None)
            pred_grounding_index = getattr(base_env, "pred_grounding_index", None)
            if not shield_cfg or shield_cfg.get("method") != "probabilistic_logic_shield" or pred_grounding_index is None:
                return
            self._pls_enabled = True
            self._pls_pred_grounding_index = copy.deepcopy(pred_grounding_index)
            self._shield_cfg = copy.deepcopy(shield_cfg)
            self._safety_coefficient = float(self._shield_cfg.get("safety_coefficient", self._shield_cfg.get("alpha", 0.1)))
            stop_action = int(getattr(self.action_space, "n", 4) - 1)
            self._shield = ProbabilisticLogicShield(
                self._pls_pred_grounding_index,
                num_actions=int(getattr(self.action_space, "n", 4)),
                stop_action=stop_action,
                sensor_uncertainty=self._shield_cfg.get("sensor_uncertainty"),
                use_observation_probabilities=bool(self._shield_cfg.get("use_observation_probabilities", False)),
            )
        except Exception:
            self._pls_enabled = False
            self._pls_pred_grounding_index = None
            self._shield_cfg = None
            self._shield = None

    def _ensure_shield(self) -> None:
        if self._shield is None and self._pls_enabled and self._pls_pred_grounding_index is not None:
            stop_action = int(getattr(self.action_space, "n", 4) - 1)
            self._shield = ProbabilisticLogicShield(
                self._pls_pred_grounding_index,
                num_actions=int(getattr(self.action_space, "n", 4)),
                stop_action=stop_action,
                sensor_uncertainty=(self._shield_cfg or {}).get("sensor_uncertainty"),
                use_observation_probabilities=bool((self._shield_cfg or {}).get("use_observation_probabilities", False)),
            )

    def configure_shield(self, pred_grounding_index, shield_cfg=None) -> None:
        self._pls_enabled = True
        self._pls_pred_grounding_index = copy.deepcopy(pred_grounding_index)
        if shield_cfg is not None:
            self._shield_cfg = copy.deepcopy(shield_cfg)
        self._safety_coefficient = float((self._shield_cfg or {}).get("safety_coefficient", (self._shield_cfg or {}).get("alpha", 0.1)))
        stop_action = int(getattr(self.action_space, "n", 4) - 1)
        self._shield = ProbabilisticLogicShield(
            self._pls_pred_grounding_index,
            num_actions=int(getattr(self.action_space, "n", 4)),
            stop_action=stop_action,
            sensor_uncertainty=(self._shield_cfg or {}).get("sensor_uncertainty"),
            use_observation_probabilities=bool((self._shield_cfg or {}).get("use_observation_probabilities", False)),
        )

    def reset_shield_metrics(self) -> None:
        self._ensure_shield()
        if self._shield is not None:
            self._shield.reset_metrics()

    def get_shield_metrics(self) -> dict[str, float]:
        self._ensure_shield()
        if self._shield is None:
            return {
                "shield_intervention_count": 0,
                "shield_intervention_rate": 0.0,
            }
        return self._shield.get_metrics()

    def _safe_probs_tensor(self, observation: np.ndarray, base_probs: th.Tensor, track_metrics: bool = False) -> th.Tensor:
        self._ensure_shield()
        if (not self._pls_enabled) or self._shield is None:
            return th.ones_like(base_probs)
        obs_np = np.asarray(observation, dtype=np.float32)
        if obs_np.ndim == 1:
            safe = self._shield.safety_probs(obs_np) if track_metrics else self._shield.safety_probs_no_metrics(obs_np)
            safe = np.expand_dims(safe, axis=0)
        else:
            safe = np.stack(
                [
                    self._shield.safety_probs(obs_np[idx]) if track_metrics else self._shield.safety_probs_no_metrics(obs_np[idx])
                    for idx in range(obs_np.shape[0])
                ],
                axis=0,
            )
        return th.as_tensor(safe, device=base_probs.device, dtype=base_probs.dtype)

    def _shield_probs_tensor(self, observation: np.ndarray, base_probs: th.Tensor, track_metrics: bool = False) -> th.Tensor:
        safe_probs = self._safe_probs_tensor(observation, base_probs, track_metrics=track_metrics)
        weighted = base_probs * safe_probs
        denom = weighted.sum(dim=1, keepdim=True)
        fallback = th.zeros_like(weighted)
        fallback[:, -1] = 1.0
        normalized = weighted / denom.clamp(min=1e-8)
        return th.where(denom > 1e-8, normalized, fallback)

    def _safety_probability_tensor(self, observation: np.ndarray, shielded_probs: th.Tensor) -> th.Tensor:
        safe_probs = self._safe_probs_tensor(observation, shielded_probs, track_metrics=False)
        return (shielded_probs * safe_probs).sum(dim=1).clamp(min=1e-8, max=1.0)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        self._ensure_shield()
        if (not self._pls_enabled) or self._shield is None:
            return super().predict(observation, state=state, episode_start=episode_start, deterministic=deterministic)

        obs_tensor = obs_as_tensor(observation, self.device)
        with th.no_grad():
            distribution = self.policy.get_distribution(obs_tensor)
            base_dist = getattr(distribution, "distribution", distribution)
            base_probs = base_dist.probs
            shielded_probs = self._shield_probs_tensor(observation, base_probs, track_metrics=True)

        if deterministic:
            actions_tensor = th.argmax(shielded_probs, dim=1)
        else:
            actions_tensor = th.distributions.Categorical(probs=shielded_probs).sample()

        actions = actions_tensor.cpu().numpy()
        if np.asarray(observation).ndim == len(self.observation_space.shape):
            actions = actions.squeeze(axis=0)
        return actions, state

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        assert self._last_obs is not None, "No previous observation was provided"
        assert isinstance(self.action_space, spaces.Discrete), "PLSPPO currently supports discrete action spaces only."

        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        self.reset_shield_metrics()

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        n_steps = 0
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                distribution = self.policy.get_distribution(obs_tensor)
                base_dist = getattr(distribution, "distribution", distribution)
                base_probs = base_dist.probs
                shielded_probs = self._shield_probs_tensor(self._last_obs, base_probs, track_metrics=True)
                actions_tensor = th.distributions.Categorical(probs=shielded_probs).sample()
                chosen_probs = shielded_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                log_probs = th.log(chosen_probs + 1e-8)
                values = self.policy.predict_values(obs_tensor)

            actions = actions_tensor.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            rollout_buffer.add(
                self._last_obs,
                actions.reshape(-1, 1),
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
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses, clip_fractions, safety_losses = [], [], [], []
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                obs_tensor = rollout_data.observations
                obs_np = obs_tensor.detach().cpu().numpy()

                distribution = self.policy.get_distribution(obs_tensor)
                base_dist = getattr(distribution, "distribution", distribution)
                base_probs = base_dist.probs
                shielded_probs = self._shield_probs_tensor(obs_np, base_probs, track_metrics=False)
                shielded_dist = th.distributions.Categorical(probs=shielded_probs)
                log_prob = shielded_dist.log_prob(actions)
                entropy = shielded_dist.entropy()
                values = self.policy.predict_values(obs_tensor).flatten()

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

                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # PLPG safety term: -alpha * log P_{pi+}(safe | s)
                safety_prob = self._safety_probability_tensor(obs_np, shielded_probs)
                safety_loss = -th.log(safety_prob).mean()
                safety_losses.append(safety_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self._safety_coefficient * safety_loss

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
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs) if len(approx_kl_divs) > 0 else 0.0)
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        self.logger.record("train/safety_coefficient", self._safety_coefficient)
