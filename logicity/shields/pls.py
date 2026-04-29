import math
import logging
from collections import OrderedDict
from typing import Any

import numpy as np
from logicity.utils.pred_converter.z3 import IsSafeStep1, IsSafeStep2, IsSafeStep3, IsSafeWait

try:
    from problog import get_evaluatable
    from problog.logic import Term
    from problog.program import PrologString
except ImportError:  # pragma: no cover - optional dependency
    PrologString = None
    get_evaluatable = None
    Term = None

# ProbLog emits very verbose INFO logs for every evaluation. Keep those quiet
# during training/evaluation unless the application explicitly re-enables them.
logging.getLogger("problog").setLevel(logging.WARNING)


class _BaseProbabilisticLogicShield:
    def __init__(
        self,
        pred_grounding_index: dict[str, tuple[int, int]],
        num_actions: int = 4,
        stop_action: int = 3,
        epsilon: float = 1e-6,
        use_privileged_internal_safety: bool = False,
        stop_priority_scale: float = 0.2,
        graded_safety: bool = False,
        graded_safety_weights: dict[str, dict[str, float]] | None = None,
    ):
        self.pred_grounding_index = pred_grounding_index
        self.num_actions = int(num_actions)
        self.stop_action = int(stop_action)
        self.epsilon = float(epsilon)
        self.use_privileged_internal_safety = bool(use_privileged_internal_safety)
        self.intervention_count = 0
        self.total_calls = 0
        self.uses_action_safe_obs = self._detect_action_safe_obs()
        self.uses_compact_relational_obs = self._detect_compact_relational_obs()
        self.n_entities = self._infer_num_entities()
        self.stop_priority_scale = float(stop_priority_scale)
        self.graded_safety = bool(graded_safety)
        self.action_names = {
            0: "slow",
            1: "normal",
            2: "fast",
            int(self.stop_action): "stop",
        }
        self.graded_safety_weights = self._normalize_graded_safety_weights(graded_safety_weights)
        self._privileged_action_safe_cache: OrderedDict[tuple[float, ...], dict[str, float]] = OrderedDict()
        self._privileged_cache_limit = 8192

    def _default_graded_safety_weights(self) -> dict[str, dict[str, float]]:
        return {
            "must_stop": {
                "slow": 0.05,
                "normal": 0.01,
                "fast": 0.0,
                "stop": 1.0,
            },
            "warning": {
                "slow": 0.95,
                "normal": 0.55,
                "fast": 0.10,
                "stop": 1.0,
            },
            "fast_zone": {
                "slow": 0.65,
                "normal": 0.85,
                "fast": 0.98,
                "stop": 1.0,
            },
            "normal_zone": {
                "slow": 0.80,
                "normal": 0.97,
                "fast": 0.25,
                "stop": 1.0,
            },
        }

    def _normalize_graded_safety_weights(
        self,
        graded_safety_weights: dict[str, dict[str, float]] | None,
    ) -> dict[str, dict[str, float]]:
        default = self._default_graded_safety_weights()
        if graded_safety_weights is None:
            return default
        merged: dict[str, dict[str, float]] = {}
        for regime_name, default_regime in default.items():
            regime_override = graded_safety_weights.get(regime_name, {}) if isinstance(graded_safety_weights, dict) else {}
            merged[regime_name] = {}
            for action_name, default_value in default_regime.items():
                override_value = regime_override.get(action_name, default_value) if isinstance(regime_override, dict) else default_value
                merged[regime_name][action_name] = float(np.clip(override_value, 0.0, 1.0))
        return merged

    def _detect_action_safe_obs(self) -> bool:
        return any(
            pred_name in self.pred_grounding_index
            for pred_name in ("IsSafeStep1", "IsSafeStep2", "IsSafeStep3", "IsSafeWait")
        )

    def _detect_compact_relational_obs(self) -> bool:
        required = {"IsAtInter", "IsInInter", "HigherPri", "CollidingCloseStep1", "CollidingCloseStep2", "CollidingCloseStep3"}
        return required.issubset(self.pred_grounding_index.keys())

    def _infer_num_entities(self) -> int:
        for pred_name in ("IsSafeStep1", "IsSafeStep2", "IsSafeStep3", "IsSafeWait"):
            if pred_name in self.pred_grounding_index:
                start, end = self.pred_grounding_index[pred_name]
                return int(end - start)
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
        return float(np.clip(observed_value, 0.0, 1.0))

    def _binary_prob(self, obs: np.ndarray, pred_name: str, entity_i: int, entity_j: int, action_name: str) -> float:
        start, _ = self.pred_grounding_index[pred_name]
        offset = entity_i * self.n_entities + entity_j
        observed_value = float(obs[start + offset])
        return float(np.clip(observed_value, 0.0, 1.0))

    def _action_safe_prob(self, obs: np.ndarray, pred_name: str) -> float:
        if pred_name not in self.pred_grounding_index:
            return 0.0
        start, _ = self.pred_grounding_index[pred_name]
        observed_value = float(obs[start])
        return float(np.clip(observed_value, 0.0, 1.0))

    def _unary_values(self, obs: np.ndarray, pred_name: str) -> np.ndarray:
        if pred_name not in self.pred_grounding_index:
            return np.zeros(self.n_entities, dtype=np.float32)
        start, end = self.pred_grounding_index[pred_name]
        values = np.asarray(obs[start:end], dtype=np.float32)
        if values.shape[0] < self.n_entities:
            padded = np.zeros(self.n_entities, dtype=np.float32)
            padded[: values.shape[0]] = values
            return padded
        return values[: self.n_entities]

    def _ego_centered_binary_values(self, obs: np.ndarray, pred_name: str) -> np.ndarray:
        if pred_name not in self.pred_grounding_index:
            return np.zeros(max(self.n_entities - 1, 0), dtype=np.float32)
        start, end = self.pred_grounding_index[pred_name]
        return np.asarray(obs[start:end], dtype=np.float32)

    def _prefer_safe_motion(self, safe_probs: np.ndarray) -> np.ndarray:
        adjusted = np.asarray(safe_probs, dtype=np.float32).copy()
        # Paper-faithful runs should not inject an extra anti-stop bias on top of
        # the logical safety model. Setting stop_priority_scale >= 1 disables the
        # heuristic entirely.
        if self.stop_priority_scale >= 1.0:
            return adjusted
        move_indices = [idx for idx in range(self.num_actions) if idx != self.stop_action]
        if not move_indices:
            return adjusted
        max_move = float(np.max(adjusted[move_indices]))
        if max_move <= 0.0:
            return adjusted
        adjusted[self.stop_action] = min(
            float(adjusted[self.stop_action]),
            float(self.stop_priority_scale * max_move),
        )
        return adjusted

    def _compact_relational_safe_probs(self, obs: Any) -> np.ndarray:
        obs_vec = self._prepare_obs(obs)
        ego_at_inter = float(self._unary_values(obs_vec, "IsAtInter")[0] > 0.5)
        ego_in_inter = float(self._unary_values(obs_vec, "IsInInter")[0] > 0.5)
        other_at_inter = self._unary_values(obs_vec, "IsAtInter")[1:self.n_entities]
        other_in_inter = self._unary_values(obs_vec, "IsInInter")[1:self.n_entities]
        higher_pri = self._ego_centered_binary_values(obs_vec, "HigherPri")
        colliding_step1 = self._ego_centered_binary_values(obs_vec, "CollidingCloseStep1")
        colliding_step2 = self._ego_centered_binary_values(obs_vec, "CollidingCloseStep2")
        colliding_step3 = self._ego_centered_binary_values(obs_vec, "CollidingCloseStep3")

        def _step_safe(conflict_values: np.ndarray) -> float:
            direct_conflict = float(np.any(conflict_values > 0.5))
            if ego_in_inter > 0.5:
                return float(not direct_conflict)
            wait_for_inside = float(np.any(other_in_inter > 0.5)) if ego_at_inter > 0.5 else 0.0
            wait_for_priority = float(np.any((other_at_inter > 0.5) & (higher_pri > 0.5))) if ego_at_inter > 0.5 else 0.0
            unsafe = (direct_conflict > 0.5) or (wait_for_inside > 0.5) or (wait_for_priority > 0.5)
            return float(not unsafe)

        safe = np.array(
            [
                _step_safe(colliding_step1),
                _step_safe(colliding_step2),
                _step_safe(colliding_step3),
                1.0,
            ],
            dtype=np.float32,
        )
        return self._prefer_safe_motion(safe)

    @staticmethod
    def _obs_cache_key(obs: Any) -> tuple[float, ...]:
        obs_vec = np.asarray(obs, dtype=np.float32).reshape(-1)
        return tuple(np.round(obs_vec, 6).tolist())

    @staticmethod
    def _cache_get(cache: OrderedDict, key: Any) -> Any:
        value = cache.get(key)
        if value is not None:
            cache.move_to_end(key)
        return value

    @staticmethod
    def _cache_put(cache: OrderedDict, key: Any, value: Any, limit: int) -> None:
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > limit:
            cache.popitem(last=False)

    def register_privileged_context(self, obs: Any, context: dict[str, Any] | None) -> None:
        if not self.use_privileged_internal_safety:
            return
        if context is None:
            return
        try:
            world_matrix = context["world_matrix"]
            intersect_matrix = context["intersect_matrix"]
            agents = context["agents"]
            ego_entity = context["ego_entity"]
        except KeyError:
            return
        safe_probs = {
            "obs_safe_slow": float(IsSafeStep1(world_matrix, intersect_matrix, agents, ego_entity) > 0.5),
            "obs_safe_normal": float(IsSafeStep2(world_matrix, intersect_matrix, agents, ego_entity) > 0.5),
            "obs_safe_fast": float(IsSafeStep3(world_matrix, intersect_matrix, agents, ego_entity) > 0.5),
            "obs_safe_stop": float(IsSafeWait(world_matrix, intersect_matrix, agents, ego_entity) > 0.5),
        }
        self._cache_put(self._privileged_action_safe_cache, self._obs_cache_key(obs), safe_probs, self._privileged_cache_limit)

    def _privileged_action_safe_probs(self, obs: Any) -> dict[str, float] | None:
        return self._cache_get(self._privileged_action_safe_cache, self._obs_cache_key(obs))

    def debug_state_facts(self, obs: Any) -> dict[str, float]:
        return {}


class _HeuristicProbabilisticLogicShield(_BaseProbabilisticLogicShield):
    def __init__(
        self,
        pred_grounding_index: dict[str, tuple[int, int]],
        num_actions: int = 4,
        stop_action: int = 3,
        epsilon: float = 1e-3,
        stop_safety: float = 0.97,
        risk_deadzone: float = 0.15,
        action_risk_weights: dict[int, float] | None = None,
    ):
        super().__init__(
            pred_grounding_index=pred_grounding_index,
            num_actions=num_actions,
            stop_action=stop_action,
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
        if self.uses_action_safe_obs:
            if track_metrics:
                self.total_calls += 1
            obs_vec = self._prepare_obs(obs)
            safe = np.array(
                [
                    self._action_safe_prob(obs_vec, "IsSafeStep1"),
                    self._action_safe_prob(obs_vec, "IsSafeStep2"),
                    self._action_safe_prob(obs_vec, "IsSafeStep3"),
                    self._action_safe_prob(obs_vec, "IsSafeWait"),
                ],
                dtype=np.float32,
            )
            safe = self._prefer_safe_motion(safe)
            if track_metrics and np.any(safe[:-1] < (1.0 - self.epsilon)):
                self.intervention_count += 1
            return safe
        if self.uses_compact_relational_obs:
            if track_metrics:
                self.total_calls += 1
            safe = self._compact_relational_safe_probs(obs)
            if track_metrics and np.any(safe[:-1] < (1.0 - self.epsilon)):
                self.intervention_count += 1
            return safe
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

    def debug_state_facts(self, obs: Any) -> dict[str, float]:
        obs_vec = self._prepare_obs(obs)
        facts: dict[str, float] = {}
        if self.uses_action_safe_obs:
            for pred_name, fact_name in (
                ("IsSafeStep1", "obs_safe_slow"),
                ("IsSafeStep2", "obs_safe_normal"),
                ("IsSafeStep3", "obs_safe_fast"),
                ("IsSafeWait", "obs_safe_stop"),
            ):
                if pred_name in self.pred_grounding_index:
                    facts[fact_name] = self._action_safe_prob(obs_vec, pred_name)
            return facts
        if self.uses_compact_relational_obs:
            facts["ego_at_inter"] = float(self._unary_values(obs_vec, "IsAtInter")[0] > 0.5)
            facts["ego_in_inter"] = float(self._unary_values(obs_vec, "IsInInter")[0] > 0.5)
            other_at_inter = self._unary_values(obs_vec, "IsAtInter")[1:self.n_entities]
            other_in_inter = self._unary_values(obs_vec, "IsInInter")[1:self.n_entities]
            higher_pri = self._ego_centered_binary_values(obs_vec, "HigherPri")
            colliding_step1 = self._ego_centered_binary_values(obs_vec, "CollidingCloseStep1")
            colliding_step2 = self._ego_centered_binary_values(obs_vec, "CollidingCloseStep2")
            colliding_step3 = self._ego_centered_binary_values(obs_vec, "CollidingCloseStep3")
            for other_idx in range(1, self.n_entities):
                rel_idx = other_idx - 1
                facts[f"other_at_inter_{other_idx}"] = float(other_at_inter[rel_idx] > 0.5) if rel_idx < len(other_at_inter) else 0.0
                facts[f"other_in_inter_{other_idx}"] = float(other_in_inter[rel_idx] > 0.5) if rel_idx < len(other_in_inter) else 0.0
                facts[f"higher_pri_{other_idx}"] = float(higher_pri[rel_idx] > 0.5) if rel_idx < len(higher_pri) else 0.0
                facts[f"colliding_step1_{other_idx}"] = float(colliding_step1[rel_idx] > 0.5) if rel_idx < len(colliding_step1) else 0.0
                facts[f"colliding_step2_{other_idx}"] = float(colliding_step2[rel_idx] > 0.5) if rel_idx < len(colliding_step2) else 0.0
                facts[f"colliding_step3_{other_idx}"] = float(colliding_step3[rel_idx] > 0.5) if rel_idx < len(colliding_step3) else 0.0
            return facts
        facts["ego_at_inter"] = self._unary_prob(obs_vec, "IsAtInter", 0, "normal")
        for other_idx in range(1, self.n_entities):
            facts[f"higher_pri_{other_idx}"] = self._binary_prob(obs_vec, "HigherPri", other_idx, 0, "normal")
            facts[f"other_in_inter_{other_idx}"] = self._unary_prob(obs_vec, "IsInInter", other_idx, "normal")
            facts[f"other_at_inter_{other_idx}"] = self._unary_prob(obs_vec, "IsAtInter", other_idx, "normal")
            facts[f"colliding_close_{other_idx}"] = self._binary_prob(obs_vec, "CollidingClose", 0, other_idx, "normal")
            if "IsCloseAhead" in self.pred_grounding_index:
                facts[f"close_ahead_{other_idx}"] = self._binary_prob(obs_vec, "IsCloseAhead", 0, other_idx, "normal")
        return facts


class _ProbLogProbabilisticLogicShield(_BaseProbabilisticLogicShield):
    def __init__(self, *args, **kwargs):
        if PrologString is None or get_evaluatable is None or Term is None:
            raise ImportError(
                "ProbLog backend requested, but the `problog` package is not installed."
            )
        super().__init__(*args, **kwargs)
        self._state_prob_cache: OrderedDict[tuple[float, ...], dict[str, float]] = OrderedDict()
        self._compiled_eval_cache: OrderedDict[tuple[tuple[str, float], ...], dict[str, float]] = OrderedDict()
        self._state_cache_limit = 4096
        self._compiled_cache_limit = 16384
        self._compiled_program = self._compile_program()
        self._weight_terms = self._build_weight_terms()

    @staticmethod
    def _format_prob(probability: float) -> str:
        return f"{float(np.clip(probability, 0.0, 1.0)):.10f}"

    def _policy_probs(self, base_probs: Any) -> np.ndarray:
        probs = np.asarray(base_probs, dtype=np.float64).reshape(-1).copy()
        probs = np.clip(probs, 0.0, None)
        total = float(np.sum(probs, dtype=np.float64))
        if total <= 0.0:
            probs = np.zeros(self.num_actions, dtype=np.float64)
            probs[self.stop_action] = 1.0
            return probs
        probs = probs / total
        if probs.shape[0] != self.num_actions:
            raise ValueError(f"Expected {self.num_actions} action probabilities, got {probs.shape[0]}.")

        # Make the AD numerically stable for ProbLog. Keep a tiny slack so the
        # annotated disjunction sum stays strictly below 1.0 under evaluator checks.
        slack = 1e-9
        head_sum = float(np.sum(probs[:-1], dtype=np.float64))
        probs[-1] = max(0.0, 1.0 - head_sum - slack)
        probs = np.clip(probs, 0.0, 1.0)
        total = float(np.sum(probs, dtype=np.float64))
        if total <= 0.0:
            probs = np.zeros(self.num_actions, dtype=np.float64)
            probs[self.stop_action] = 1.0
        else:
            probs = probs / total
            head_sum = float(np.sum(probs[:-1], dtype=np.float64))
            probs[-1] = max(0.0, 1.0 - head_sum - slack)
        return probs

    def _state_probabilities(self, obs: Any) -> dict[str, float]:
        obs_key = self._obs_cache_key(obs)
        cached = self._cache_get(self._state_prob_cache, obs_key)
        if cached is not None:
            return cached
        privileged = self._privileged_action_safe_probs(obs)
        if privileged is not None:
            self._cache_put(self._state_prob_cache, obs_key, privileged, self._state_cache_limit)
            return privileged
        obs_vec = self._prepare_obs(obs)
        if self.uses_action_safe_obs:
            probs = {
                "obs_safe_slow": self._action_safe_prob(obs_vec, "IsSafeStep1"),
                "obs_safe_normal": self._action_safe_prob(obs_vec, "IsSafeStep2"),
                "obs_safe_fast": self._action_safe_prob(obs_vec, "IsSafeStep3"),
                "obs_safe_stop": self._action_safe_prob(obs_vec, "IsSafeWait"),
            }
        elif self.uses_compact_relational_obs:
            probs = {
                "ego_at_inter": float(self._unary_values(obs_vec, "IsAtInter")[0] > 0.5),
                "ego_in_inter": float(self._unary_values(obs_vec, "IsInInter")[0] > 0.5),
            }
            other_at_inter = self._unary_values(obs_vec, "IsAtInter")[1:self.n_entities]
            other_in_inter = self._unary_values(obs_vec, "IsInInter")[1:self.n_entities]
            higher_pri = self._ego_centered_binary_values(obs_vec, "HigherPri")
            colliding_step1 = self._ego_centered_binary_values(obs_vec, "CollidingCloseStep1")
            colliding_step2 = self._ego_centered_binary_values(obs_vec, "CollidingCloseStep2")
            colliding_step3 = self._ego_centered_binary_values(obs_vec, "CollidingCloseStep3")
            for other_idx in range(1, self.n_entities):
                rel_idx = other_idx - 1
                probs[f"other_at_inter_{other_idx}"] = float(other_at_inter[rel_idx] > 0.5) if rel_idx < len(other_at_inter) else 0.0
                probs[f"other_in_inter_{other_idx}"] = float(other_in_inter[rel_idx] > 0.5) if rel_idx < len(other_in_inter) else 0.0
                probs[f"higher_pri_{other_idx}"] = float(higher_pri[rel_idx] > 0.5) if rel_idx < len(higher_pri) else 0.0
                probs[f"colliding_step1_{other_idx}"] = float(colliding_step1[rel_idx] > 0.5) if rel_idx < len(colliding_step1) else 0.0
                probs[f"colliding_step2_{other_idx}"] = float(colliding_step2[rel_idx] > 0.5) if rel_idx < len(colliding_step2) else 0.0
                probs[f"colliding_step3_{other_idx}"] = float(colliding_step3[rel_idx] > 0.5) if rel_idx < len(colliding_step3) else 0.0
        else:
            probs = {
                "ego_at_inter": self._unary_prob(obs_vec, "IsAtInter", 0, "normal"),
                "ego_in_inter": self._unary_prob(obs_vec, "IsInInter", 0, "normal"),
            }
            for other_idx in range(1, self.n_entities):
                probs[f"higher_pri_{other_idx}"] = self._binary_prob(obs_vec, "HigherPri", other_idx, 0, "normal")
                probs[f"other_in_inter_{other_idx}"] = self._unary_prob(obs_vec, "IsInInter", other_idx, "normal")
                probs[f"other_at_inter_{other_idx}"] = self._unary_prob(obs_vec, "IsAtInter", other_idx, "normal")
                probs[f"colliding_close_{other_idx}"] = self._binary_prob(obs_vec, "CollidingClose", 0, other_idx, "normal")
                probs[f"close_ahead_{other_idx}"] = self._binary_prob(obs_vec, "IsCloseAhead", 0, other_idx, "normal")
        self._cache_put(self._state_prob_cache, obs_key, probs, self._state_cache_limit)
        return probs

    def _program_atom_names(self) -> list[str]:
        if self.uses_action_safe_obs:
            return [
                "obs_safe_slow",
                "obs_safe_normal",
                "obs_safe_fast",
                "obs_safe_stop",
            ]
        if self.uses_compact_relational_obs:
            atom_names = ["ego_at_inter", "ego_in_inter"]
            for other_idx in range(1, self.n_entities):
                atom_names.extend(
                    [
                        f"other_at_inter_{other_idx}",
                        f"other_in_inter_{other_idx}",
                        f"higher_pri_{other_idx}",
                        f"colliding_step1_{other_idx}",
                        f"colliding_step2_{other_idx}",
                        f"colliding_step3_{other_idx}",
                    ]
                )
            return atom_names
        atom_names = ["ego_at_inter", "ego_in_inter"]
        for other_idx in range(1, self.n_entities):
            atom_names.extend(
                [
                    f"higher_pri_{other_idx}",
                    f"other_in_inter_{other_idx}",
                    f"other_at_inter_{other_idx}",
                    f"colliding_close_{other_idx}",
                    f"close_ahead_{other_idx}",
                ]
            )
        return atom_names

    def _action_weight_names(self) -> list[str]:
        return [self.action_names[idx] for idx in range(self.num_actions)]

    def _build_program_template(self) -> str:
        lines = []
        action_labels = [self.action_names[idx] for idx in range(self.num_actions)]
        action_heads = "; ".join(f"0.25::act({label})" for label in action_labels)
        lines.append(f"{action_heads}.")
        for atom_name in self._program_atom_names():
            lines.append(f"0.5::{atom_name}.")

        if self.uses_action_safe_obs:
            lines.extend(
                [
                    "safe_action_slow :- obs_safe_slow.",
                    "safe_action_normal :- obs_safe_normal.",
                    "safe_action_fast :- obs_safe_fast.",
                    "safe_action_stop :- obs_safe_stop.",
                ]
            )
        elif self.uses_compact_relational_obs:
            for other_idx in range(1, self.n_entities):
                lines.append(
                    f"priority_conflict_{other_idx} :- ego_at_inter, higher_pri_{other_idx}, other_in_inter_{other_idx}."
                )
                lines.append(
                    f"waiting_conflict_{other_idx} :- ego_at_inter, higher_pri_{other_idx}, other_at_inter_{other_idx}."
                )
                lines.append(f"unsafe_slow :- colliding_step1_{other_idx}.")
                lines.append(
                    f"unsafe_slow :- priority_conflict_{other_idx}."
                )
                lines.append(
                    f"unsafe_slow :- waiting_conflict_{other_idx}."
                )
                lines.append(f"unsafe_normal :- colliding_step2_{other_idx}.")
                lines.append(
                    f"unsafe_normal :- priority_conflict_{other_idx}."
                )
                lines.append(
                    f"unsafe_normal :- waiting_conflict_{other_idx}."
                )
                lines.append(f"unsafe_fast :- colliding_step3_{other_idx}.")
                lines.append(
                    f"unsafe_fast :- priority_conflict_{other_idx}."
                )
                lines.append(
                    f"unsafe_fast :- waiting_conflict_{other_idx}."
                )
            lines.extend(
                [
                    "safe_action_stop.",
                    "safe_action_slow :- \\+ unsafe_slow.",
                    "safe_action_normal :- \\+ unsafe_normal.",
                    "safe_action_fast :- \\+ unsafe_fast.",
                ]
            )
        else:
            for other_idx in range(1, self.n_entities):
                lines.append(
                    f"occupancy_conflict_{other_idx} :- ego_at_inter, other_in_inter_{other_idx}."
                )
                lines.append(
                    f"waiting_conflict_{other_idx} :- ego_at_inter, higher_pri_{other_idx}, other_at_inter_{other_idx}."
                )
                lines.append(
                    f"close_conflict_{other_idx} :- colliding_close_{other_idx}."
                )
                lines.append(
                    f"warning_close_{other_idx} :- close_ahead_{other_idx}."
                )
                lines.append(f"must_stop :- occupancy_conflict_{other_idx}.")
                lines.append(f"must_stop :- waiting_conflict_{other_idx}.")
                lines.append(f"must_stop :- close_conflict_{other_idx}.")
                lines.append(f"has_close_ahead :- warning_close_{other_idx}.")
            if self.graded_safety:
                weights = self.graded_safety_weights
                lines.extend(
                    [
                        "warning_state :- \\+ must_stop, has_close_ahead.",
                        "fast_zone_state :- \\+ must_stop, \\+ has_close_ahead, \\+ ego_at_inter, \\+ ego_in_inter.",
                        "normal_zone_state :- \\+ must_stop, \\+ has_close_ahead, \\+ fast_zone_state.",
                    ]
                )
                regime_atoms = {
                    "must_stop": "must_stop",
                    "warning": "warning_state",
                    "fast_zone": "fast_zone_state",
                    "normal_zone": "normal_zone_state",
                }
                for regime_name, state_atom in regime_atoms.items():
                    for action_name in ("slow", "normal", "fast", "stop"):
                        prob = self._format_prob(weights[regime_name][action_name])
                        fact_name = f"graded_{regime_name}_{action_name}"
                        lines.append(f"{prob}::{fact_name}.")
                        lines.append(f"safe_action_{action_name} :- {state_atom}, {fact_name}.")
            else:
                lines.extend(
                    [
                        "safe_action_stop :- must_stop.",
                        "safe_action_slow :- \\+ must_stop, has_close_ahead.",
                        "safe_action_fast :- \\+ must_stop, \\+ has_close_ahead, \\+ ego_at_inter, \\+ ego_in_inter.",
                        "safe_action_normal :- \\+ must_stop, \\+ has_close_ahead, \\+ safe_action_fast.",
                    ]
                )
        lines.extend(
            [
                "safe :- act(slow), safe_action_slow.",
                "safe :- act(normal), safe_action_normal.",
                "safe :- act(fast), safe_action_fast.",
                "safe :- act(stop), safe_action_stop.",
                "joint_safe_slow :- act(slow), safe.",
                "joint_safe_normal :- act(normal), safe.",
                "joint_safe_fast :- act(fast), safe.",
                "joint_safe_stop :- act(stop), safe.",
                "query(act(slow)).",
                "query(act(normal)).",
                "query(act(fast)).",
                "query(act(stop)).",
                "query(safe_action_slow).",
                "query(safe_action_normal).",
                "query(safe_action_fast).",
                "query(safe_action_stop).",
                "query(safe).",
                "query(joint_safe_slow).",
                "query(joint_safe_normal).",
                "query(joint_safe_fast).",
                "query(joint_safe_stop).",
            ]
        )
        return "\n".join(lines)

    def _compile_program(self):
        program = self._build_program_template()
        return get_evaluatable().create_from(PrologString(program))

    def _build_weight_terms(self) -> dict[str, Any]:
        weight_terms = {atom_name: Term(atom_name) for atom_name in self._program_atom_names()}
        for action_name in self._action_weight_names():
            weight_terms[f"act_{action_name}"] = Term("act", Term(action_name))
        return weight_terms

    def _evaluate_program(self, state_probs: dict[str, float], policy_probs: np.ndarray) -> dict[str, float]:
        cache_entries = list(sorted((name, round(probability, 6)) for name, probability in state_probs.items()))
        for idx, action_name in self.action_names.items():
            cache_entries.append((f"act_{action_name}", round(float(policy_probs[idx]), 6)))
        cache_key = tuple(cache_entries)
        cached = self._cache_get(self._compiled_eval_cache, cache_key)
        if cached is not None:
            return cached
        weights = {
            self._weight_terms[name]: float(np.clip(probability, 0.0, 1.0))
            for name, probability in state_probs.items()
            if name in self._weight_terms
        }
        for idx, action_name in self.action_names.items():
            term_key = f"act_{action_name}"
            if term_key not in self._weight_terms:
                # Checkpoints created before the joint-policy patch may restore a
                # shield instance without the action weight entries. Rebuild the
                # weight-term table on demand so loaded models still work.
                self._weight_terms = self._build_weight_terms()
            weights[self._weight_terms[term_key]] = float(np.clip(policy_probs[idx], 0.0, 1.0))
        result = self._compiled_program.evaluate(weights=weights)
        parsed = {str(key): float(value) for key, value in result.items()}
        self._cache_put(self._compiled_eval_cache, cache_key, parsed, self._compiled_cache_limit)
        return parsed

    def _policy_from_input(self, policy_probs: Any | None) -> np.ndarray:
        if policy_probs is None:
            probs = np.ones(self.num_actions, dtype=np.float32) / float(self.num_actions)
            return probs
        return self._policy_probs(policy_probs)

    def _result_for_obs_policy(self, obs: Any, policy_probs: Any | None) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
        probs = self._policy_from_input(policy_probs)
        state_probs = self._state_probabilities(obs)
        result = self._evaluate_program(state_probs, probs)
        return probs, state_probs, result

    def _safe_action_probs_from_result(self, result: dict[str, float], policy_probs: np.ndarray) -> np.ndarray:
        joint = np.array(
            [
                float(result.get("joint_safe_slow", 0.0)),
                float(result.get("joint_safe_normal", 0.0)),
                float(result.get("joint_safe_fast", 0.0)),
                float(result.get("joint_safe_stop", 0.0)),
            ],
            dtype=np.float32,
        )
        safe = np.zeros(self.num_actions, dtype=np.float32)
        for idx in range(self.num_actions):
            denom = float(policy_probs[idx])
            if denom > self.epsilon:
                safe[idx] = float(np.clip(joint[idx] / denom, 0.0, 1.0))
            else:
                safe[idx] = float(result.get(f"safe_action_{self.action_names[idx]}", 0.0))
        return safe

    def _compute_safety_probs(self, obs: Any, policy_probs: Any | None, track_metrics: bool) -> np.ndarray:
        if track_metrics:
            self.total_calls += 1
        probs, _, result = self._result_for_obs_policy(obs, policy_probs)
        safe = self._safe_action_probs_from_result(result, probs)
        safe = self._prefer_safe_motion(safe)
        if track_metrics and np.any(safe[:-1] < (1.0 - self.epsilon)):
            self.intervention_count += 1
        return safe

    def safety_probs(self, obs: Any, policy_probs: Any | None = None) -> np.ndarray:
        return self._compute_safety_probs(obs, policy_probs, track_metrics=True)

    def safety_probs_no_metrics(self, obs: Any, policy_probs: Any | None = None) -> np.ndarray:
        return self._compute_safety_probs(obs, policy_probs, track_metrics=False)

    def policy_safety_probability(self, obs: Any, policy_probs: Any) -> float:
        probs, _, result = self._result_for_obs_policy(obs, policy_probs)
        safe_probability = float(result.get("safe", 0.0))
        if self.stop_priority_scale != 1.0:
            safe_probs = self._prefer_safe_motion(self._safe_action_probs_from_result(result, probs))
            safe_probability = float(np.sum(probs * safe_probs))
        return float(np.clip(safe_probability, self.epsilon, 1.0))

    def shield_probs(self, obs: Any, base_probs: Any, track_metrics: bool = True) -> np.ndarray:
        if track_metrics:
            self.total_calls += 1
        policy_probs, _, result = self._result_for_obs_policy(obs, base_probs)
        safe_action_probs = self._safe_action_probs_from_result(result, policy_probs)
        adjusted_safe = self._prefer_safe_motion(safe_action_probs)
        weighted = policy_probs * adjusted_safe
        weighted_total = float(np.sum(weighted))
        if weighted_total <= self.epsilon:
            shielded = np.zeros_like(policy_probs, dtype=np.float32)
            shielded[self.stop_action] = 1.0
        else:
            shielded = weighted / weighted_total
        if track_metrics and np.max(np.abs(shielded - policy_probs)) > 1e-6:
            self.intervention_count += 1
        return shielded

    def shield_probs_no_metrics(self, obs: Any, base_probs: Any) -> np.ndarray:
        return self.shield_probs(obs, base_probs, track_metrics=False)

    def debug_state_facts(self, obs: Any) -> dict[str, float]:
        return dict(self._state_probabilities(obs))


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
        epsilon: float = 1e-3,
        stop_safety: float = 0.97,
        risk_deadzone: float = 0.15,
        action_risk_weights: dict[int, float] | None = None,
        backend: str = "heuristic",
        use_privileged_internal_safety: bool = False,
        stop_priority_scale: float = 0.2,
        graded_safety: bool = False,
        graded_safety_weights: dict[str, dict[str, float]] | None = None,
    ):
        backend_name = str(backend).lower()
        common_kwargs = {
            "pred_grounding_index": pred_grounding_index,
            "num_actions": num_actions,
            "stop_action": stop_action,
            "epsilon": epsilon,
            "use_privileged_internal_safety": use_privileged_internal_safety,
            "stop_priority_scale": stop_priority_scale,
            "graded_safety": graded_safety,
            "graded_safety_weights": graded_safety_weights,
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

    def register_privileged_context(self, obs: Any, context: dict[str, Any] | None) -> None:
        self._impl.register_privileged_context(obs, context)

    def safety_probs(self, obs: Any, policy_probs: Any | None = None) -> np.ndarray:
        return self._impl.safety_probs(obs, policy_probs)

    def safety_probs_no_metrics(self, obs: Any, policy_probs: Any | None = None) -> np.ndarray:
        return self._impl.safety_probs_no_metrics(obs, policy_probs)

    def policy_safety_probability(self, obs: Any, policy_probs: Any) -> float:
        return self._impl.policy_safety_probability(obs, policy_probs)

    def shield_probs(self, obs: Any, base_probs: Any, track_metrics: bool = True) -> np.ndarray:
        return self._impl.shield_probs(obs, base_probs, track_metrics=track_metrics)

    def shield_probs_no_metrics(self, obs: Any, base_probs: Any) -> np.ndarray:
        return self._impl.shield_probs_no_metrics(obs, base_probs)

    def debug_state_facts(self, obs: Any) -> dict[str, float]:
        return self._impl.debug_state_facts(obs)
