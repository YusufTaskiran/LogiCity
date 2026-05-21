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


class JointPLSPPO(SB3PPO):
    """Shared-policy two-car PPO with a joint ProbLog shield during training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._joint_pls_enabled = False
        self._shield_cfg = None
        self._joint_selector = None
        self._safety_coefficient = 0.1
        self._use_safety_loss = True
        self._action_labels = None
        self._action_label_by_index = None
        self._stop_action = int(getattr(self.action_space, "n", 4) - 1)
        self._diagnostics = {}
        self._rollout_joint_contexts = None
        self._init_joint_pls_from_env()

    def _init_joint_pls_from_env(self):
        if self.env is None:
            return
        shield_cfg = getattr(self.env, "shield_config", None)
        if not shield_cfg or shield_cfg.get("method") != "probabilistic_logic_shield":
            return
        self._joint_pls_enabled = True
        self._shield_cfg = copy.deepcopy(shield_cfg)
        self._safety_coefficient = float(self._shield_cfg.get("safety_coefficient", self._shield_cfg.get("alpha", 0.1)))
        self._use_safety_loss = bool(self._shield_cfg.get("use_safety_loss", True))
        labels = list(self._shield_cfg.get("action_space", ["fast", "normal", "slow", "stop"]))
        if len(labels) != int(getattr(self.action_space, "n", 4)):
            raise ValueError("Shield action_space labels do not match discrete action space size.")
        self._action_labels = [str(v).strip().lower() for v in labels]
        self._action_label_by_index = {idx: label for idx, label in enumerate(self._action_labels)}
        self._stop_action = int(self._action_labels.index("stop"))
        self._joint_selector = JointProbabilisticLogicShieldTwoCar(
            num_actions=int(getattr(self.action_space, "n", 4)),
            stop_action=self._stop_action,
            action_space=self._action_labels,
            graded_safety_weights=self._shield_cfg.get("graded_safety_weights"),
        )

    def reset_joint_shield_metrics(self):
        if self._joint_selector is not None:
            self._joint_selector.reset_metrics()
        self._diagnostics = {
            "joint_rollout_samples": 0.0,
            "joint_pair_safety_sum": 0.0,
            "train_pair_safety_sum": 0.0,
            "train_batches": 0.0,
        }

    def get_joint_shield_metrics(self) -> dict[str, float]:
        metrics = {"joint_shield_intervention_count": 0, "joint_shield_intervention_rate": 0.0}
        if self._joint_selector is not None:
            metrics.update(self._joint_selector.get_metrics())
        samples = float(self._diagnostics.get("joint_rollout_samples", 0.0)) if self._diagnostics else 0.0
        if samples > 0.0:
            metrics["joint_rollout_pair_safety_mean"] = self._diagnostics["joint_pair_safety_sum"] / samples
        batches = float(self._diagnostics.get("train_batches", 0.0)) if self._diagnostics else 0.0
        if batches > 0.0:
            metrics["train_pair_safety_mean"] = self._diagnostics["train_pair_safety_sum"] / batches
        return metrics

    def _regime_weight_vector(self, regime_name: str) -> th.Tensor:
        values = [
            float(self._shield_cfg["graded_safety_weights"][regime_name][self._action_label_by_index[idx]])
            for idx in range(self.action_space.n)
        ]
        return th.as_tensor(values, device=self.device, dtype=th.float32)

    def _joint_safe_scores_tensor(
        self,
        admissible_mask: th.Tensor,
        both_at_inter: th.Tensor,
        same_priority: th.Tensor,
        tie_break_a: th.Tensor,
    ) -> th.Tensor:
        normal_w = self._regime_weight_vector("normal_zone")
        warning_w = self._regime_weight_vector("warning")
        must_stop_w = self._regime_weight_vector("must_stop")

        score_away = normal_w.view(1, -1, 1) * normal_w.view(1, 1, -1)
        score_a_moves = warning_w.view(1, -1, 1) * must_stop_w.view(1, 1, -1)
        score_b_moves = must_stop_w.view(1, -1, 1) * warning_w.view(1, 1, -1)

        move_mask = th.ones(self.action_space.n, device=self.device, dtype=th.float32)
        move_mask[self._stop_action] = 0.0
        stop_mask = th.zeros(self.action_space.n, device=self.device, dtype=th.float32)
        stop_mask[self._stop_action] = 1.0

        a_moves_mask = move_mask.view(1, -1, 1) * stop_mask.view(1, 1, -1)
        b_moves_mask = stop_mask.view(1, -1, 1) * move_mask.view(1, 1, -1)

        score_both_at_inter = th.where(
            same_priority.view(-1, 1, 1) > 0.5,
            th.where(tie_break_a.view(-1, 1, 1) > 0.5, score_a_moves * a_moves_mask, score_b_moves * b_moves_mask),
            (score_a_moves * a_moves_mask) + (score_b_moves * b_moves_mask),
        )
        safe_scores = th.where(
            both_at_inter.view(-1, 1, 1) > 0.5,
            score_both_at_inter,
            score_away,
        )
        safe_scores = safe_scores * admissible_mask
        return safe_scores.clamp(0.0, 1.0)

    def _joint_pair_distribution_tensor(
        self,
        base_probs_pair: th.Tensor,
        admissible_mask: th.Tensor,
        both_at_inter: th.Tensor,
        same_priority: th.Tensor,
        tie_break_a: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        safe_scores = self._joint_safe_scores_tensor(admissible_mask, both_at_inter, same_priority, tie_break_a)
        joint_base = base_probs_pair[:, 0, :].unsqueeze(2) * base_probs_pair[:, 1, :].unsqueeze(1)
        weighted = joint_base * safe_scores
        denom = weighted.sum(dim=(1, 2), keepdim=True)
        fallback = th.zeros_like(weighted)
        fallback[:, self._stop_action, self._stop_action] = 1.0
        pair_dist = th.where(denom > 1e-8, weighted / denom.clamp_min(1e-8), fallback)
        return pair_dist, safe_scores

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        assert self._last_obs is not None, "No previous observation was provided"
        assert isinstance(self.action_space, spaces.Discrete), "JointPLSPPO currently supports discrete action spaces only."
        if not hasattr(env, "current_joint_context"):
            raise ValueError("JointPLSPPO requires a JointTwoCarVecEnv-like environment with current_joint_context().")

        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        self.reset_joint_shield_metrics()
        self._rollout_joint_contexts = {
            "admissible_mask": np.zeros((n_rollout_steps, self.action_space.n, self.action_space.n), dtype=np.float32),
            "both_at_inter": np.zeros((n_rollout_steps,), dtype=np.float32),
            "same_priority": np.zeros((n_rollout_steps,), dtype=np.float32),
            "tie_break_a": np.zeros((n_rollout_steps,), dtype=np.float32),
        }

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        n_steps = 0
        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                distribution = self.policy.get_distribution(obs_tensor)
                base_dist = getattr(distribution, "distribution", distribution)
                base_probs = base_dist.probs
                values = self.policy.predict_values(obs_tensor).reshape(-1)

                ctx = env.current_joint_context()
                admissible_mask = th.as_tensor(ctx["admissible_mask"], device=self.device, dtype=base_probs.dtype).unsqueeze(0)
                both_at_inter = th.as_tensor([float(ctx["both_at_inter"])], device=self.device, dtype=base_probs.dtype)
                same_priority = th.as_tensor([float(ctx["same_priority"])], device=self.device, dtype=base_probs.dtype)
                tie_break_a = th.as_tensor([float(ctx["tie_break_a"])], device=self.device, dtype=base_probs.dtype)

                pair_dist, safe_scores = self._joint_pair_distribution_tensor(
                    base_probs_pair=base_probs.unsqueeze(0),
                    admissible_mask=admissible_mask,
                    both_at_inter=both_at_inter,
                    same_priority=same_priority,
                    tie_break_a=tie_break_a,
                )
                flat_dist = pair_dist.reshape(-1)
                chosen_flat = th.distributions.Categorical(probs=flat_dist).sample()
                a_idx = int(chosen_flat.item() // self.action_space.n)
                b_idx = int(chosen_flat.item() % self.action_space.n)
                actions_tensor = th.as_tensor([a_idx, b_idx], device=self.device, dtype=th.long)
                marginals = th.stack([pair_dist[0].sum(dim=1), pair_dist[0].sum(dim=0)], dim=0)
                chosen_probs = marginals.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                log_probs = th.log(chosen_probs.clamp_min(1e-8))

                self._diagnostics["joint_rollout_samples"] += 1.0
                self._diagnostics["joint_pair_safety_sum"] += float((pair_dist * safe_scores).sum().detach().cpu().item())

            actions = actions_tensor.detach().cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)

            self._rollout_joint_contexts["admissible_mask"][n_steps] = ctx["admissible_mask"]
            self._rollout_joint_contexts["both_at_inter"][n_steps] = float(ctx["both_at_inter"])
            self._rollout_joint_contexts["same_priority"][n_steps] = float(ctx["same_priority"])
            self._rollout_joint_contexts["tie_break_a"][n_steps] = float(ctx["tie_break_a"])

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
        time_batch_size = max(1, int(self.batch_size // env_count))
        indices = np.arange(buffer_size)

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, buffer_size, time_batch_size):
                mb_idx = indices[start : start + time_batch_size]

                obs_pair = th.as_tensor(self.rollout_buffer.observations[mb_idx], device=self.device, dtype=th.float32)
                actions_pair = th.as_tensor(self.rollout_buffer.actions[mb_idx], device=self.device).long().squeeze(-1)
                old_values_pair = th.as_tensor(self.rollout_buffer.values[mb_idx], device=self.device, dtype=th.float32)
                old_log_probs_pair = th.as_tensor(self.rollout_buffer.log_probs[mb_idx], device=self.device, dtype=th.float32)
                advantages_pair = th.as_tensor(self.rollout_buffer.advantages[mb_idx], device=self.device, dtype=th.float32)
                returns_pair = th.as_tensor(self.rollout_buffer.returns[mb_idx], device=self.device, dtype=th.float32)

                obs_flat = obs_pair.reshape(-1, obs_pair.shape[-1])
                distribution = self.policy.get_distribution(obs_flat)
                base_dist = getattr(distribution, "distribution", distribution)
                base_probs = base_dist.probs.reshape(-1, env_count, self.action_space.n)
                values_pair = self.policy.predict_values(obs_flat).reshape(-1, env_count)

                admissible_mask = th.as_tensor(self._rollout_joint_contexts["admissible_mask"][mb_idx], device=self.device, dtype=base_probs.dtype)
                both_at_inter = th.as_tensor(self._rollout_joint_contexts["both_at_inter"][mb_idx], device=self.device, dtype=base_probs.dtype)
                same_priority = th.as_tensor(self._rollout_joint_contexts["same_priority"][mb_idx], device=self.device, dtype=base_probs.dtype)
                tie_break_a = th.as_tensor(self._rollout_joint_contexts["tie_break_a"][mb_idx], device=self.device, dtype=base_probs.dtype)

                pair_dist, safe_scores = self._joint_pair_distribution_tensor(
                    base_probs_pair=base_probs,
                    admissible_mask=admissible_mask,
                    both_at_inter=both_at_inter,
                    same_priority=same_priority,
                    tie_break_a=tie_break_a,
                )
                marginals = th.stack([pair_dist.sum(dim=2), pair_dist.sum(dim=1)], dim=1)
                new_log_probs_pair = th.log(
                    marginals.gather(2, actions_pair.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
                )
                entropy_pair = -(marginals.clamp_min(1e-8) * th.log(marginals.clamp_min(1e-8))).sum(dim=2)

                advantages = advantages_pair.reshape(-1)
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                log_prob = new_log_probs_pair.reshape(-1)
                old_log_prob = old_log_probs_pair.reshape(-1)
                ratio = th.exp(log_prob - old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                values = values_pair.reshape(-1)
                old_values = old_values_pair.reshape(-1)
                returns = returns_pair.reshape(-1)
                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = old_values + th.clamp(values - old_values, -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(returns, values_pred)
                value_losses.append(value_loss.item())

                entropy_loss = -th.mean(entropy_pair.reshape(-1))
                entropy_losses.append(entropy_loss.item())

                if self._use_safety_loss:
                    pair_safety_prob = (pair_dist * safe_scores).sum(dim=(1, 2)).clamp(1e-8, 1.0)
                    safety_loss = -th.log(pair_safety_prob).mean()
                    self._diagnostics["train_pair_safety_sum"] += float(pair_safety_prob.mean().detach().cpu().item())
                    self._diagnostics["train_batches"] += 1.0
                else:
                    safety_loss = th.zeros((), device=self.device, dtype=pair_dist.dtype)
                safety_losses.append(safety_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self._safety_coefficient * safety_loss

                with th.no_grad():
                    log_ratio = log_prob - old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at epoch {epoch} due to reaching max kl: {approx_kl_div:.2f}")
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
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        self.logger.record("train/safety_coefficient", self._safety_coefficient)
        self.logger.record("train/use_safety_loss", float(self._use_safety_loss))
        for key, value in self.get_joint_shield_metrics().items():
            self.logger.record(f"train/{key}", value)

