import math
from typing import Any

import numpy as np

from .sensor_model import PredicateSensorModel


class DeterministicLogicShield:
    """Deterministic probability-level logic shield.

    In the current thesis setup the discrete action space is:
    0 = slow, 1 = normal, 2 = fast, 3 = stop.

    The shield computes a hard safety mask P(safe | s, a) in {0, 1} and
    renormalizes the base policy over safe actions only.
    """

    def __init__(
        self,
        pred_grounding_index: dict[str, tuple[int, int]],
        num_actions: int = 4,
        safe_action: int = 3,
        sensor_uncertainty: dict[str, Any] | None = None,
        use_observation_probabilities: bool = False,
    ):
        self.pred_grounding_index = pred_grounding_index
        self.num_actions = int(num_actions)
        self.safe_action = int(safe_action)
        self.sensor_model = PredicateSensorModel(sensor_uncertainty)
        self.use_observation_probabilities = bool(use_observation_probabilities)
        self.intervention_count = 0
        self.total_calls = 0
        self.n_entities = self._infer_num_entities()
        self.action_names = {
            0: "slow",
            1: "normal",
            2: "fast",
            int(self.safe_action): "stop",
        }

    def _infer_num_entities(self) -> int:
        # Unary predicates ground once per entity.
        for pred_name in ("IsCar", "IsPedestrian", "IsAtInter", "IsInInter"):
            if pred_name in self.pred_grounding_index:
                start, end = self.pred_grounding_index[pred_name]
                return int(end - start)
        # Binary fallback.
        if "HigherPri" in self.pred_grounding_index:
            start, end = self.pred_grounding_index["HigherPri"]
            return int(round(math.sqrt(end - start)))
        raise ValueError("Could not infer entity count from pred_grounding_index.")

    def reset_metrics(self) -> None:
        self.intervention_count = 0
        self.total_calls = 0

    def get_metrics(self) -> dict[str, float]:
        rate = float(self.intervention_count / self.total_calls) if self.total_calls > 0 else 0.0
        return {
            "shield_intervention_count": int(self.intervention_count),
            "shield_intervention_rate": rate,
        }

    def _prepare_obs(self, obs: Any) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 1:
            return arr
        raise ValueError(f"Expected a single observation vector, got shape {arr.shape}.")

    def _unary(self, obs: np.ndarray, pred_name: str, entity_idx: int, action_name: str) -> bool:
        start, _ = self.pred_grounding_index[pred_name]
        observed_value = float(obs[start + entity_idx])
        if self.use_observation_probabilities:
            return self.sensor_model.threshold_probability(observed_value)
        truth_value = bool(observed_value > 0.5)
        sensed_prob = self.sensor_model.sense_probability(pred_name, truth_value, action_name=action_name)
        return self.sensor_model.threshold_probability(sensed_prob)

    def _binary(self, obs: np.ndarray, pred_name: str, entity_i: int, entity_j: int, action_name: str) -> bool:
        start, _ = self.pred_grounding_index[pred_name]
        offset = entity_i * self.n_entities + entity_j
        observed_value = float(obs[start + offset])
        if self.use_observation_probabilities:
            return self.sensor_model.threshold_probability(observed_value)
        truth_value = bool(observed_value > 0.5)
        sensed_prob = self.sensor_model.sense_probability(pred_name, truth_value, action_name=action_name)
        return self.sensor_model.threshold_probability(sensed_prob)

    def stop_required(self, obs: Any, action_name: str) -> bool:
        return self.risk_band(obs, action_name) == "high"

    def risk_score(self, obs: Any, action_name: str) -> float:
        obs_vec = self._prepare_obs(obs)
        ego_at_inter = self._unary(obs_vec, "IsAtInter", 0, action_name)
        per_other_conflicts: list[float] = []
        for other_idx in range(1, self.n_entities):
            other_has_priority = self._binary(obs_vec, "HigherPri", other_idx, 0, action_name)
            if not other_has_priority:
                continue
            other_in_inter = self._unary(obs_vec, "IsInInter", other_idx, action_name)
            other_at_inter = self._unary(obs_vec, "IsAtInter", other_idx, action_name)
            colliding_close = self._binary(obs_vec, "CollidingClose", 0, other_idx, action_name)
            conflict = float((ego_at_inter and (other_in_inter or other_at_inter)) or colliding_close)
            if conflict > 0.0:
                per_other_conflicts.append(conflict)
        if not per_other_conflicts:
            return 0.0
        return float(sum(per_other_conflicts) / len(per_other_conflicts))

    def risk_band(self, obs: Any, action_name: str) -> str:
        risk = self.risk_score(obs, action_name)
        if risk >= 0.95:
            return "high"
        if risk >= 0.45:
            return "medium"
        return "low"

    def safety_probs(self, obs: Any) -> np.ndarray:
        self.total_calls += 1
        safe_probs = np.ones(self.num_actions, dtype=np.float32)
        masked_any = False
        risk_bands = {}
        for action_idx in range(self.num_actions):
            if action_idx == self.safe_action:
                safe_probs[action_idx] = 1.0
                continue
            action_name = self.action_names.get(action_idx, "normal")
            risk_bands[action_idx] = self.risk_band(obs, action_name)

        # Tiered deterministic shield:
        # low risk -> allow normal, slow, stop
        # medium risk -> allow slow, stop
        # high risk -> allow stop only
        normal_idx = 1 if self.num_actions > 1 else 0
        fast_idx = 2 if self.num_actions > 2 else normal_idx
        low_risk_fast = risk_bands.get(fast_idx) == "low"

        for action_idx in range(self.num_actions):
            if action_idx == self.safe_action:
                continue
            band = risk_bands.get(action_idx, "low")
            allowed = True
            if action_idx == 0:  # slow
                allowed = band in ("low", "medium")
            elif action_idx == normal_idx:  # normal
                allowed = band == "low"
            elif action_idx == fast_idx:  # fast
                allowed = low_risk_fast and risk_bands.get(normal_idx, "low") == "low"
            if not allowed:
                safe_probs[action_idx] = 0.0
                masked_any = True
        if masked_any:
            self.intervention_count += 1
        return safe_probs

    def shield_probs(self, obs: Any, base_probs: Any) -> np.ndarray:
        base_probs = np.asarray(base_probs, dtype=np.float32)
        safe_probs = self.safety_probs(obs)
        weighted = base_probs * safe_probs
        total = float(np.sum(weighted))
        if total <= 0:
            fallback = np.zeros_like(base_probs, dtype=np.float32)
            fallback[self.safe_action] = 1.0
            return fallback
        return weighted / total

    def shield_action(self, obs: Any, action: int) -> int:
        self.total_calls += 1
        action = int(action)
        action_name = self.action_names.get(action, "normal")
        if action != self.safe_action and self.stop_required(obs, action_name):
            self.intervention_count += 1
            return self.safe_action
        return action
