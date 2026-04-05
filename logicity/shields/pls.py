import math
from typing import Any

import numpy as np

from .sensor_model import PredicateSensorModel


class ProbabilisticLogicShield:
    """Probabilistic logic shield for the 4-action thesis task.

    Actions:
    0 = slow, 1 = normal, 2 = fast, 3 = stop

    The shield computes a soft safety vector P(safe | s, a) and renormalizes
    the base policy accordingly:

        pi_plus(a|s) propto P(safe|s,a) * pi(a|s)
    """

    def __init__(
        self,
        pred_grounding_index: dict[str, tuple[int, int]],
        num_actions: int = 4,
        stop_action: int = 3,
        sensor_uncertainty: dict[str, Any] | None = None,
        use_observation_probabilities: bool = False,
        epsilon: float = 1e-3,
        stop_safety: float = 0.97,
        risk_deadzone: float = 0.15,
        action_risk_weights: dict[int, float] | None = None,
    ):
        self.pred_grounding_index = pred_grounding_index
        self.num_actions = int(num_actions)
        self.stop_action = int(stop_action)
        self.sensor_model = PredicateSensorModel(sensor_uncertainty)
        self.use_observation_probabilities = bool(use_observation_probabilities)
        self.epsilon = float(epsilon)
        self.stop_safety = float(stop_safety)
        self.risk_deadzone = float(risk_deadzone)
        self.action_risk_weights = action_risk_weights or {
            0: 0.2,   # slow
            1: 0.4,   # normal
            2: 0.7,   # fast
            3: 0.0,   # stop
        }
        self.intervention_count = 0
        self.total_calls = 0
        self.n_entities = self._infer_num_entities()
        self.action_names = {
            0: "slow",
            1: "normal",
            2: "fast",
            int(self.stop_action): "stop",
        }

    def _infer_num_entities(self) -> int:
        for pred_name in ("IsCar", "IsPedestrian", "IsAtInter", "IsInInter"):
            if pred_name in self.pred_grounding_index:
                start, end = self.pred_grounding_index[pred_name]
                return int(end - start)
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

    def _unary_prob(self, obs: np.ndarray, pred_name: str, entity_idx: int, action_name: str) -> float:
        start, _ = self.pred_grounding_index[pred_name]
        observed_value = float(obs[start + entity_idx])
        if self.use_observation_probabilities:
            return observed_value
        truth_value = bool(observed_value > 0.5)
        return float(self.sensor_model.sense_probability(pred_name, truth_value, action_name=action_name))

    def _binary_prob(self, obs: np.ndarray, pred_name: str, entity_i: int, entity_j: int, action_name: str) -> float:
        start, _ = self.pred_grounding_index[pred_name]
        offset = entity_i * self.n_entities + entity_j
        observed_value = float(obs[start + offset])
        if self.use_observation_probabilities:
            return observed_value
        truth_value = bool(observed_value > 0.5)
        return float(self.sensor_model.sense_probability(pred_name, truth_value, action_name=action_name))

    def conflict_probability(self, obs: Any, action_name: str) -> float:
        obs_vec = self._prepare_obs(obs)
        ego_at_inter = self._unary_prob(obs_vec, "IsAtInter", 0, action_name)
        per_other_conflicts = []
        for other_idx in range(1, self.n_entities):
            other_has_priority = self._binary_prob(obs_vec, "HigherPri", other_idx, 0, action_name)
            if other_has_priority <= 0.0:
                continue
            other_in_inter = self._unary_prob(obs_vec, "IsInInter", other_idx, action_name)
            other_at_inter = self._unary_prob(obs_vec, "IsAtInter", other_idx, action_name)
            colliding_close = self._binary_prob(obs_vec, "CollidingClose", 0, other_idx, action_name)
            c1 = ego_at_inter * other_in_inter * other_has_priority
            c2 = ego_at_inter * other_at_inter * other_has_priority
            c3 = colliding_close * other_has_priority
            per_other_conflicts.append(1.0 - ((1.0 - c1) * (1.0 - c2) * (1.0 - c3)))
        if not per_other_conflicts:
            return 0.0
        risk = 1.0
        for conflict in per_other_conflicts:
            risk *= (1.0 - float(conflict))
        return float(1.0 - risk)

    def safety_probs(self, obs: Any) -> np.ndarray:
        self.total_calls += 1
        safe = np.ones(self.num_actions, dtype=np.float32)
        intervention = False
        risk_by_action = {}
        for action_idx in range(self.num_actions):
            action_name = self.action_names.get(action_idx, "normal")
            if action_idx == self.stop_action:
                continue
            risk = self.conflict_probability(obs, action_name)
            if risk < self.risk_deadzone:
                risk = 0.0
            risk_by_action[action_idx] = risk
            if risk > 0.0:
                intervention = True

        slow_risk = float(risk_by_action.get(0, 0.0))
        normal_risk = float(risk_by_action.get(1, slow_risk))
        fast_risk = float(risk_by_action.get(2, normal_risk))

        # Make normal the preferred action in the broad middle regime:
        # very low risk -> fast best
        # low/moderate risk -> normal best
        # elevated risk -> slow best
        # high risk -> stop best
        safe[0] = float(np.clip(0.92 - 0.55 * slow_risk + 0.12 * normal_risk, 0.18, 0.95))
        safe[1] = float(np.clip(1.03 - 0.75 * normal_risk, 0.28, 1.00))
        safe[2] = float(np.clip(1.08 - 1.35 * fast_risk, 0.03, 1.00))

        if fast_risk > 0.20:
            safe[2] *= 0.70
        if normal_risk > 0.45:
            safe[1] *= 0.82
        if slow_risk > 0.70:
            safe[0] *= 0.72

        combined_risk = max(slow_risk, normal_risk, fast_risk)
        safe[self.stop_action] = float(np.clip(0.18 + 1.05 * combined_risk, 0.18, self.stop_safety))
        if intervention:
            self.intervention_count += 1
        return safe

    def shield_probs(self, obs: Any, base_probs: Any) -> np.ndarray:
        base_probs = np.asarray(base_probs, dtype=np.float32)
        safe_probs = self.safety_probs(obs)
        weighted = base_probs * safe_probs
        total = float(np.sum(weighted))
        if total <= 0:
            fallback = np.zeros_like(base_probs, dtype=np.float32)
            fallback[self.stop_action] = 1.0
            return fallback
        return weighted / total
