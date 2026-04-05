import copy
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO as SB3PPO
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

    def _shield_probs_tensor(self, observation: np.ndarray, base_probs: th.Tensor) -> th.Tensor:
        self._ensure_shield()
        if (not self._pls_enabled) or self._shield is None:
            return base_probs

        base_probs_np = base_probs.detach().cpu().numpy()
        obs_np = np.asarray(observation, dtype=np.float32)
        if obs_np.ndim == 1:
            shielded = self._shield.shield_probs(obs_np, base_probs_np[0])
            shielded = np.expand_dims(shielded, axis=0)
        else:
            shielded = np.stack(
                [self._shield.shield_probs(obs_np[idx], base_probs_np[idx]) for idx in range(obs_np.shape[0])],
                axis=0,
            )
        return th.as_tensor(shielded, device=base_probs.device, dtype=base_probs.dtype)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        self._ensure_shield()
        if (not self._pls_enabled) or self._shield is None:
            return super().predict(observation, state=state, episode_start=episode_start, deterministic=deterministic)

        obs_tensor = obs_as_tensor(observation, self.device)
        with th.no_grad():
            distribution = self.policy.get_distribution(obs_tensor)
            base_dist = getattr(distribution, "distribution", distribution)
            base_probs = base_dist.probs
            shielded_probs = self._shield_probs_tensor(observation, base_probs)

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
                shielded_probs = self._shield_probs_tensor(self._last_obs, base_probs)
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
