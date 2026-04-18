import math
from typing import Any

import numpy as np

from .sensor_model import PredicateSensorModel

try:
    from problog import get_evaluatable
    from problog.program import PrologString
except ImportError:  # pragma: no cover - optional dependency
    PrologString = None
    get_evaluatable = None


class _BaseProbabilisticLogicShield:
    def __init__(
        self,
        pred_grounding_index: dict[str, tuple[int, int]],
        num_actions: int = 4,
        stop_action: int = 3,
        sensor_uncertainty: dict[str, Any] | None = None,
        use_observation_probabilities: bool = False,
        epsilon: float = 1e-6,
    ):
        self.pred_grounding_index = pred_grounding_index
        self.num_actions = int(num_actions)
        self.stop_action = int(stop_action)
        self.sensor_model = PredicateSensorModel(sensor_uncertainty)
        self.use_observation_probabilities = bool(use_observation_probabilities)
        self.epsilon = float(epsilon)
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
            return float(np.clip(observed_value, 0.0, 1.0))
        truth_value = bool(observed_value > 0.5)
        return float(self.sensor_model.sense_probability(pred_name, truth_value, action_name=action_name))

    def _binary_prob(self, obs: np.ndarray, pred_name: str, entity_i: int, entity_j: int, action_name: str) -> float:
        start, _ = self.pred_grounding_index[pred_name]
        offset = entity_i * self.n_entities + entity_j
        observed_value = float(obs[start + offset])
        if self.use_observation_probabilities:
            return float(np.clip(observed_value, 0.0, 1.0))
        truth_value = bool(observed_value > 0.5)
        return float(self.sensor_model.sense_probability(pred_name, truth_value, action_name=action_name))


class _HeuristicProbabilisticLogicShield(_BaseProbabilisticLogicShield):
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
        super().__init__(
            pred_grounding_index=pred_grounding_index,
            num_actions=num_actions,
            stop_action=stop_action,
            sensor_uncertainty=sensor_uncertainty,
            use_observation_probabilities=use_observation_probabilities,
            epsilon=epsilon,
        )
        self.stop_safety = float(stop_safety)
        self.risk_deadzone = float(risk_deadzone)
        self.action_risk_weights = action_risk_weights or {
            0: 0.2,
            1: 0.4,
            2: 0.7,
            3: 0.0,
        }

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

    def _compute_safety_probs(self, obs: Any, track_metrics: bool) -> np.ndarray:
        if track_metrics:
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
        if intervention and track_metrics:
            self.intervention_count += 1
        return safe

    def safety_probs(self, obs: Any) -> np.ndarray:
        return self._compute_safety_probs(obs, track_metrics=True)

    def safety_probs_no_metrics(self, obs: Any) -> np.ndarray:
        return self._compute_safety_probs(obs, track_metrics=False)

    def policy_safety_probability(self, obs: Any, policy_probs: Any) -> float:
        safe_probs = self.safety_probs_no_metrics(obs)
        policy_probs = np.asarray(policy_probs, dtype=np.float32)
        return float(np.clip(np.sum(policy_probs * safe_probs), self.epsilon, 1.0))

    def shield_probs(self, obs: Any, base_probs: Any, track_metrics: bool = True) -> np.ndarray:
        base_probs = np.asarray(base_probs, dtype=np.float32)
        safe_probs = self.safety_probs(obs) if track_metrics else self.safety_probs_no_metrics(obs)
        weighted = base_probs * safe_probs
        total = float(np.sum(weighted))
        if total <= 0:
            fallback = np.zeros_like(base_probs, dtype=np.float32)
            fallback[self.stop_action] = 1.0
            return fallback
        return weighted / total

    def shield_probs_no_metrics(self, obs: Any, base_probs: Any) -> np.ndarray:
        return self.shield_probs(obs, base_probs, track_metrics=False)


class _ProbLogProbabilisticLogicShield(_BaseProbabilisticLogicShield):
    def __init__(self, *args, **kwargs):
        if PrologString is None or get_evaluatable is None:
            raise ImportError(
                "ProbLog backend requested, but the `problog` package is not installed."
            )
        super().__init__(*args, **kwargs)

    @staticmethod
    def _format_prob(probability: float) -> str:
        return f"{float(np.clip(probability, 0.0, 1.0)):.10f}"

    def _policy_probs(self, base_probs: Any) -> np.ndarray:
        probs = np.asarray(base_probs, dtype=np.float32).copy()
        probs = np.clip(probs, 0.0, None)
        total = float(np.sum(probs))
        if total <= 0.0:
            probs = np.zeros(self.num_actions, dtype=np.float32)
            probs[self.stop_action] = 1.0
            return probs
        probs = probs / total
        non_stop_total = float(np.sum(probs) - probs[self.stop_action])
        probs[self.stop_action] = max(0.0, 1.0 - non_stop_total)
        probs = np.clip(probs, 0.0, 1.0)
        return probs / float(max(np.sum(probs), self.epsilon))

    def _state_probabilities(self, obs: Any) -> dict[str, float]:
        obs_vec = self._prepare_obs(obs)
        probs: dict[str, float] = {
            "ego_at_inter": self._unary_prob(obs_vec, "IsAtInter", 0, "normal"),
        }
        for other_idx in range(1, self.n_entities):
            probs[f"higher_pri_{other_idx}"] = self._binary_prob(obs_vec, "HigherPri", other_idx, 0, "normal")
            probs[f"other_in_inter_{other_idx}"] = self._unary_prob(obs_vec, "IsInInter", other_idx, "normal")
            probs[f"other_at_inter_{other_idx}"] = self._unary_prob(obs_vec, "IsAtInter", other_idx, "normal")
            probs[f"colliding_close_{other_idx}"] = self._binary_prob(obs_vec, "CollidingClose", 0, other_idx, "normal")
        return probs

    def _build_program(self, state_probs: dict[str, float], policy_probs: np.ndarray) -> str:
        lines = []
        slow_prob = float(policy_probs[0])
        normal_prob = float(policy_probs[1])
        fast_prob = float(policy_probs[2])
        stop_prob = max(0.0, 1.0 - slow_prob - normal_prob - fast_prob)
        lines.append(
            f"{self._format_prob(slow_prob)}::act(slow); "
            f"{self._format_prob(normal_prob)}::act(normal); "
            f"{self._format_prob(fast_prob)}::act(fast); "
            f"{self._format_prob(stop_prob)}::act(stop)."
        )
        for atom_name, probability in state_probs.items():
            lines.append(f"{self._format_prob(probability)}::{atom_name}.")

        for other_idx in range(1, self.n_entities):
            lines.append(
                f"priority_conflict_{other_idx} :- ego_at_inter, higher_pri_{other_idx}, other_in_inter_{other_idx}."
            )
            lines.append(
                f"waiting_conflict_{other_idx} :- ego_at_inter, higher_pri_{other_idx}, other_at_inter_{other_idx}."
            )
            lines.append(
                f"close_conflict_{other_idx} :- colliding_close_{other_idx}, higher_pri_{other_idx}."
            )
            lines.append(f"unsafe_fast :- priority_conflict_{other_idx}.")
            lines.append(f"unsafe_fast :- waiting_conflict_{other_idx}.")
            lines.append(f"unsafe_fast :- close_conflict_{other_idx}.")
            lines.append(f"unsafe_normal :- priority_conflict_{other_idx}.")
            lines.append(f"unsafe_normal :- close_conflict_{other_idx}.")
            lines.append(f"unsafe_slow :- close_conflict_{other_idx}.")

        lines.extend(
            [
                "safe :- act(stop).",
                "safe :- act(slow), \\+ unsafe_slow.",
                "safe :- act(normal), \\+ unsafe_normal.",
                "safe :- act(fast), \\+ unsafe_fast.",
                "safe_action_slow :- act(slow), safe.",
                "safe_action_normal :- act(normal), safe.",
                "safe_action_fast :- act(fast), safe.",
                "safe_action_stop :- act(stop), safe.",
                "query(safe).",
                "query(safe_action_slow).",
                "query(safe_action_normal).",
                "query(safe_action_fast).",
                "query(safe_action_stop).",
            ]
        )
        return "\n".join(lines)

    def _evaluate_program(self, state_probs: dict[str, float], policy_probs: np.ndarray) -> dict[str, float]:
        program = self._build_program(state_probs, policy_probs)
        result = get_evaluatable().create_from(PrologString(program)).evaluate()
        return {str(key): float(value) for key, value in result.items()}

    def _query_action_safety(self, state_probs: dict[str, float], action_idx: int) -> float:
        one_hot = np.zeros(self.num_actions, dtype=np.float32)
        one_hot[action_idx] = 1.0
        result = self._evaluate_program(state_probs, one_hot)
        return float(np.clip(result.get("safe", 0.0), self.epsilon, 1.0))

    def _compute_safety_probs(self, obs: Any, track_metrics: bool) -> np.ndarray:
        if track_metrics:
            self.total_calls += 1
        state_probs = self._state_probabilities(obs)
        safe = np.zeros(self.num_actions, dtype=np.float32)
        for action_idx in range(self.num_actions):
            safe[action_idx] = self._query_action_safety(state_probs, action_idx)
        if track_metrics and np.any(safe[:-1] < (1.0 - self.epsilon)):
            self.intervention_count += 1
        return safe

    def safety_probs(self, obs: Any) -> np.ndarray:
        return self._compute_safety_probs(obs, track_metrics=True)

    def safety_probs_no_metrics(self, obs: Any) -> np.ndarray:
        return self._compute_safety_probs(obs, track_metrics=False)

    def policy_safety_probability(self, obs: Any, policy_probs: Any) -> float:
        state_probs = self._state_probabilities(obs)
        result = self._evaluate_program(state_probs, self._policy_probs(policy_probs))
        return float(np.clip(result.get("safe", 0.0), self.epsilon, 1.0))

    def shield_probs(self, obs: Any, base_probs: Any, track_metrics: bool = True) -> np.ndarray:
        if track_metrics:
            self.total_calls += 1
        policy_probs = self._policy_probs(base_probs)
        state_probs = self._state_probabilities(obs)
        result = self._evaluate_program(state_probs, policy_probs)
        safe_prob = float(result.get("safe", 0.0))
        joint = np.array(
            [
                result.get("safe_action_slow", 0.0),
                result.get("safe_action_normal", 0.0),
                result.get("safe_action_fast", 0.0),
                result.get("safe_action_stop", 0.0),
            ],
            dtype=np.float32,
        )
        if safe_prob <= self.epsilon:
            shielded = np.zeros_like(policy_probs, dtype=np.float32)
            shielded[self.stop_action] = 1.0
        else:
            shielded = joint / float(max(safe_prob, self.epsilon))
            total = float(np.sum(shielded))
            if total <= 0.0:
                shielded = np.zeros_like(policy_probs, dtype=np.float32)
                shielded[self.stop_action] = 1.0
            else:
                shielded = shielded / total
        if track_metrics and np.max(np.abs(shielded - policy_probs)) > 1e-6:
            self.intervention_count += 1
        return shielded

    def shield_probs_no_metrics(self, obs: Any, base_probs: Any) -> np.ndarray:
        return self.shield_probs(obs, base_probs, track_metrics=False)


class ProbabilisticLogicShield:
    """Probabilistic logic shield for the 4-action thesis task.

    Supported backends:
    - ``heuristic``: previous hand-crafted approximation
    - ``problog``: ProbLog program queried as in the paper-style formulation
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
        backend: str = "heuristic",
    ):
        backend_name = str(backend).lower()
        common_kwargs = {
            "pred_grounding_index": pred_grounding_index,
            "num_actions": num_actions,
            "stop_action": stop_action,
            "sensor_uncertainty": sensor_uncertainty,
            "use_observation_probabilities": use_observation_probabilities,
            "epsilon": epsilon,
        }
        if backend_name == "problog":
            self._impl = _ProbLogProbabilisticLogicShield(**common_kwargs)
        else:
            self._impl = _HeuristicProbabilisticLogicShield(
                **common_kwargs,
                stop_safety=stop_safety,
                risk_deadzone=risk_deadzone,
                action_risk_weights=action_risk_weights,
            )

    def reset_metrics(self) -> None:
        self._impl.reset_metrics()

    def get_metrics(self) -> dict[str, float]:
        return self._impl.get_metrics()

    def safety_probs(self, obs: Any) -> np.ndarray:
        return self._impl.safety_probs(obs)

    def safety_probs_no_metrics(self, obs: Any) -> np.ndarray:
        return self._impl.safety_probs_no_metrics(obs)

    def policy_safety_probability(self, obs: Any, policy_probs: Any) -> float:
        return self._impl.policy_safety_probability(obs, policy_probs)

    def shield_probs(self, obs: Any, base_probs: Any, track_metrics: bool = True) -> np.ndarray:
        return self._impl.shield_probs(obs, base_probs, track_metrics=track_metrics)

    def shield_probs_no_metrics(self, obs: Any, base_probs: Any) -> np.ndarray:
        return self._impl.shield_probs_no_metrics(obs, base_probs)
