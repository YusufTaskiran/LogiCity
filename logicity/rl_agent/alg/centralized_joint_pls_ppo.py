from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.utils import explained_variance, obs_as_tensor

from ...shields import JointProbabilisticLogicShieldTwoCar

logger = logging.getLogger(__name__)


def _unwrap_first_env(env: Any) -> Any:
    current = env
    if hasattr(current, "envs") and len(getattr(current, "envs")) > 0:
        current = current.envs[0]
    while hasattr(current, "env") and not hasattr(current, "shield_config"):
        current = current.env
    return current


class CentralizedJointPLSPPO(SB3PPO):
    """Centralized PPO over a 16-way joint action space with a joint ProbLog shield."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._joint_pls_enabled = False
        self._shield_cfg = None
        self._joint_selector = None
        self._safety_coefficient = 0.1
        self._use_safety_loss = True
        self._num_local_actions = 4
        self._stop_joint_action = 15
        self._diagnostics = {}
        self._rollout_safe_scores = None
        self._init_joint_pls_from_env()

    def _init_joint_pls_from_env(self):
        if self.env is None:
            return
        base_env = _unwrap_first_env(self.env)
        shield_cfg = getattr(base_env, "shield_config", None)
        if not shield_cfg or shield_cfg.get("method") != "probabilistic_logic_shield":
            return
        self._joint_pls_enabled = True
        self._shield_cfg = copy.deepcopy(shield_cfg)
        self._safety_coefficient = float(self._shield_cfg.get("safety_coefficient", self._shield_cfg.get("alpha", 0.1)))
        self._use_safety_loss = bool(self._shield_cfg.get("use_safety_loss", True))
        labels = list(self._shield_cfg.get("action_space", ["fast", "normal", "slow", "stop"]))
        self._num_local_actions = len(labels)
        stop_idx = labels.index("stop")
        self._stop_joint_action = stop_idx * self._num_local_actions + stop_idx
        self._joint_selector = JointProbabilisticLogicShieldTwoCar(
            num_actions=self._num_local_actions,
            stop_action=stop_idx,
            action_space=labels,
            graded_safety_weights=self._shield_cfg.get("graded_safety_weights"),
        )

    def reset_joint_shield_metrics(self):
        if self._joint_selector is not None:
            self._joint_selector.reset_metrics()
        self._diagnostics = {
            "rollout_joint_safety_sum": 0.0,
            "rollout_samples": 0.0,
            "train_joint_safety_sum": 0.0,
            "train_batches": 0.0,
        }

    def get_joint_shield_metrics(self) -> dict[str, float]:
        metrics = {"joint_shield_intervention_count": 0, "joint_shield_intervention_rate": 0.0}
        if self._joint_selector is not None:
            metrics.update(self._joint_selector.get_metrics())
        samples = float(self._diagnostics.get("rollout_samples", 0.0)) if self._diagnostics else 0.0
        if samples > 0.0:
            metrics["rollout_joint_safety_mean"] = self._diagnostics["rollout_joint_safety_sum"] / samples
        batches = float(self._diagnostics.get("train_batches", 0.0)) if self._diagnostics else 0.0
        if batches > 0.0:
            metrics["train_joint_safety_mean"] = self._diagnostics["train_joint_safety_sum"] / batches
        return metrics

    def _safe_scores_tensor(self, env_source: Any, base_probs: th.Tensor, track_metrics: bool = False) -> th.Tensor:
        if (not self._joint_pls_enabled) or self._joint_selector is None:
            return th.ones_like(base_probs)
        base_env = _unwrap_first_env(env_source if env_source is not None else self.env)
        ctx = getattr(base_env, "current_shield_context", None)
        if ctx is None:
            return th.ones_like(base_probs)
        safe_scores = np.asarray(ctx["safe_scores"], dtype=np.float32).reshape(-1)
        safe_tensor = th.as_tensor(safe_scores, device=base_probs.device, dtype=base_probs.dtype)
        if base_probs.ndim == 1:
            safe_tensor = safe_tensor.unsqueeze(0)
        else:
            safe_tensor = safe_tensor.unsqueeze(0).expand(base_probs.shape[0], -1)
        if track_metrics and self._joint_selector is not None and base_probs.shape[0] > 0:
            probs_np = base_probs.detach().cpu().numpy()
            if probs_np.ndim == 1:
                probs_np = np.expand_dims(probs_np, axis=0)
            greedy_base = int(np.argmax(probs_np[0]))
            greedy_safe = int(np.argmax((probs_np[0] * safe_scores)))
            self._joint_selector.total_calls += 1
            if greedy_base != greedy_safe:
                self._joint_selector.intervention_count += 1
        return safe_tensor

    def _shield_probs_tensor(self, env_source: Any, base_probs: th.Tensor, track_metrics: bool = False) -> tuple[th.Tensor, th.Tensor]:
        if (not self._joint_pls_enabled) or self._joint_selector is None:
            return base_probs, th.ones_like(base_probs)
        squeezed = False
        probs_batched = base_probs
        if probs_batched.ndim == 1:
            probs_batched = probs_batched.unsqueeze(0)
            squeezed = True
        safe_probs = self._safe_scores_tensor(env_source, probs_batched, track_metrics=track_metrics)
        weighted = probs_batched * safe_probs
        denom = weighted.sum(dim=1, keepdim=True)
        fallback = th.zeros_like(probs_batched)
        fallback[:, self._stop_joint_action] = 1.0
        normalized = weighted / denom.clamp_min(1e-8)
        shielded = th.where(denom > 1e-8, normalized, fallback)
        if squeezed:
            return shielded.squeeze(0), safe_probs.squeeze(0)
        return shielded, safe_probs

    @staticmethod
    def _policy_safety_from_safe_probs(policy_probs: th.Tensor, safe_probs: th.Tensor) -> th.Tensor:
        return (policy_probs * safe_probs).sum(dim=1).clamp(1e-8, 1.0)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if (not self._joint_pls_enabled) or self._joint_selector is None:
            return super().predict(observation, state=state, episode_start=episode_start, deterministic=deterministic)

        obs_tensor = obs_as_tensor(observation, self.device)
        with th.no_grad():
            distribution = self.policy.get_distribution(obs_tensor)
            base_dist = getattr(distribution, "distribution", distribution)
            base_probs = base_dist.probs
            shielded_probs, _ = self._shield_probs_tensor(self.env, base_probs, track_metrics=True)

        if deterministic:
            if shielded_probs.ndim == 1:
                actions_tensor = th.argmax(shielded_probs, dim=0).unsqueeze(0)
            else:
                actions_tensor = th.argmax(shielded_probs, dim=1)
        else:
            if shielded_probs.ndim == 1:
                actions_tensor = th.distributions.Categorical(probs=shielded_probs).sample().unsqueeze(0)
            else:
                actions_tensor = th.distributions.Categorical(probs=shielded_probs).sample()
        actions = actions_tensor.cpu().numpy()
        if np.asarray(observation).ndim == len(self.observation_space.shape):
            actions = actions.squeeze(axis=0)
        return actions, state

    def debug_action_snapshot(self, observation: np.ndarray, env_source: Any | None = None) -> dict[str, Any]:
        if (not self._joint_pls_enabled) or self._joint_selector is None:
            raise RuntimeError("Centralized joint shield is not enabled for this model.")
        base_env = _unwrap_first_env(env_source if env_source is not None else self.env)
        ctx = getattr(base_env, "current_shield_context", None)
        if ctx is None:
            raise RuntimeError("No current_shield_context available on the environment.")

        obs_np = np.asarray(observation, dtype=np.float32)
        if obs_np.ndim != 1:
            raise ValueError(f"Expected a single flattened observation, got shape {obs_np.shape}.")

        with th.no_grad():
            obs_tensor = obs_as_tensor(obs_np, self.device)
            distribution = self.policy.get_distribution(obs_tensor)
            base_dist = getattr(distribution, "distribution", distribution)
            base_probs = base_dist.probs
            shielded_probs, safe_probs = self._shield_probs_tensor(base_env, base_probs, track_metrics=False)

        pair_base = base_probs.detach().cpu().numpy().reshape(-1)
        num_actions = self._num_local_actions
        base_probs_a = pair_base.reshape(num_actions, num_actions).sum(axis=1)
        base_probs_b = pair_base.reshape(num_actions, num_actions).sum(axis=0)
        shield_debug = self._joint_selector.debug_snapshot(
            base_probs_a=base_probs_a,
            base_probs_b=base_probs_b,
            candidate_infos=ctx["candidate_infos"],
            joint_facts={
                "both_at_inter": ctx["both_at_inter"],
                "same_priority": ctx["same_priority"],
            },
            tiebreak_order=ctx.get("tiebreak_order", (0, 1)),
        )
        return {
            "base_joint_probs": pair_base.tolist(),
            "shielded_joint_probs": shielded_probs.detach().cpu().numpy().reshape(-1).tolist(),
            "safe_joint_scores": safe_probs.detach().cpu().numpy().reshape(-1).tolist(),
            "shield_debug": shield_debug,
        }

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        assert self._last_obs is not None, "No previous observation was provided"
        assert isinstance(self.action_space, spaces.Discrete), "CentralizedJointPLSPPO currently supports discrete action spaces only."

        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        self.reset_joint_shield_metrics()
        self._rollout_safe_scores = np.zeros((n_rollout_steps, env.num_envs, self.action_space.n), dtype=np.float32)

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
                shielded_probs, safe_probs = self._shield_probs_tensor(env, base_probs, track_metrics=True)
                actions_tensor = th.distributions.Categorical(probs=shielded_probs).sample()
                chosen_probs = shielded_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                log_probs = th.log(chosen_probs + 1e-8)
                values = self.policy.predict_values(obs_tensor)

                self._diagnostics["rollout_joint_safety_sum"] += float(
                    self._policy_safety_from_safe_probs(shielded_probs, safe_probs).mean().detach().cpu().item()
                )
                self._diagnostics["rollout_samples"] += float(base_probs.shape[0])

            actions = actions_tensor.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)
            self._rollout_safe_scores[n_steps] = safe_probs.detach().cpu().numpy()

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
        approx_kl_divs = []
        buffer_size = self.rollout_buffer.buffer_size
        env_count = self.rollout_buffer.n_envs
        flat_size = buffer_size * env_count
        batch_size = self.batch_size or flat_size
        indices = np.arange(flat_size)

        obs_all = self.rollout_buffer.observations.reshape(flat_size, *self.rollout_buffer.observations.shape[2:])
        actions_all = self.rollout_buffer.actions.reshape(flat_size, -1)
        old_values_all = self.rollout_buffer.values.reshape(flat_size)
        old_log_probs_all = self.rollout_buffer.log_probs.reshape(flat_size)
        advantages_all = self.rollout_buffer.advantages.reshape(flat_size)
        returns_all = self.rollout_buffer.returns.reshape(flat_size)
        safe_scores_all = self._rollout_safe_scores.reshape(flat_size, self.action_space.n)

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, flat_size, batch_size):
                mb_idx = indices[start : start + batch_size]

                obs_tensor = th.as_tensor(obs_all[mb_idx], device=self.device, dtype=th.float32)
                actions = th.as_tensor(actions_all[mb_idx], device=self.device).long().flatten()
                old_values = th.as_tensor(old_values_all[mb_idx], device=self.device, dtype=th.float32)
                old_log_prob = th.as_tensor(old_log_probs_all[mb_idx], device=self.device, dtype=th.float32)
                advantages = th.as_tensor(advantages_all[mb_idx], device=self.device, dtype=th.float32)
                returns = th.as_tensor(returns_all[mb_idx], device=self.device, dtype=th.float32)
                safe_probs = th.as_tensor(safe_scores_all[mb_idx], device=self.device, dtype=th.float32)

                distribution = self.policy.get_distribution(obs_tensor)
                base_dist = getattr(distribution, "distribution", distribution)
                base_probs = base_dist.probs
                weighted = base_probs * safe_probs
                denom = weighted.sum(dim=1, keepdim=True)
                fallback = th.zeros_like(base_probs)
                fallback[:, self._stop_joint_action] = 1.0
                shielded_probs = th.where(denom > 1e-8, weighted / denom.clamp_min(1e-8), fallback)
                shielded_dist = th.distributions.Categorical(probs=shielded_probs)
                log_prob = shielded_dist.log_prob(actions)
                entropy = shielded_dist.entropy()
                values = self.policy.predict_values(obs_tensor).flatten()

                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = old_values + th.clamp(
                        values - old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(returns, values_pred)
                value_losses.append(value_loss.item())

                entropy_loss = -th.mean(entropy if entropy is not None else -log_prob)
                entropy_losses.append(entropy_loss.item())

                if self._use_safety_loss:
                    shielded_safety_prob = self._policy_safety_from_safe_probs(shielded_probs, safe_probs)
                    self._diagnostics["train_joint_safety_sum"] += float(shielded_safety_prob.mean().detach().cpu().item())
                    self._diagnostics["train_batches"] += 1.0
                    safety_loss = -th.log(shielded_safety_prob).mean()
                else:
                    safety_loss = th.zeros((), device=obs_tensor.device, dtype=shielded_probs.dtype)
                safety_losses.append(safety_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self._safety_coefficient * safety_loss

                with th.no_grad():
                    log_ratio = log_prob - old_log_prob
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
        self.logger.record("train/use_safety_loss", float(self._use_safety_loss))
        for key, value in self.get_joint_shield_metrics().items():
            self.logger.record(f"train/{key}", value)
