import logging
from collections import OrderedDict
from typing import Any

import numpy as np

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


class ProbabilisticLogicShield:
    """Current thesis ProbLog shield for the 4-action symbolic setup.

    This class intentionally keeps only the active PLS path:
    - current symbolic observation layout
    - ProbLog backend
    - graded action safety over the four regimes
    """

    def __init__(
        self,
        pred_grounding_index: dict[str, tuple[int, int]],
        num_actions: int = 4,
        stop_action: int = 3,
        epsilon: float = 1e-3,
        backend: str = "problog",
        use_privileged_internal_safety: bool = False,
        stop_priority_scale: float = 1.0,
        graded_safety: bool = True,
        graded_safety_weights: dict[str, dict[str, float]] | None = None,
        action_space: list[str] | tuple[str, ...] | None = None,
        **_: Any,
    ):
        if PrologString is None or get_evaluatable is None or Term is None:
            raise ImportError("ProbLog backend requested, but the `problog` package is not installed.")
        if str(backend).lower() != "problog":
            raise ValueError("Only the current ProbLog PLS backend is supported.")
        if not bool(graded_safety):
            raise ValueError("The current PLS implementation requires graded_safety=true.")

        self.pred_grounding_index = pred_grounding_index
        self.num_actions = int(num_actions)
        self.stop_action = int(stop_action)
        self.epsilon = float(epsilon)
        self.stop_priority_scale = float(stop_priority_scale)
        self.use_privileged_internal_safety = bool(use_privileged_internal_safety)
        self.intervention_count = 0
        self.total_calls = 0
        self.minimal_observation = self._is_minimal_observation()
        self.n_entities = self._infer_num_entities()
        self.action_names = self._resolve_action_names(action_space)
        self.graded_safety_weights = self._normalize_graded_safety_weights(graded_safety_weights)
        self._state_prob_cache: OrderedDict[tuple[float, ...], dict[str, float]] = OrderedDict()
        self._compiled_eval_cache: OrderedDict[tuple[tuple[str, float], ...], dict[str, float]] = OrderedDict()
        self._privileged_context_cache: OrderedDict[tuple[float, ...], dict[str, Any]] = OrderedDict()
        self._state_cache_limit = 4096
        self._compiled_cache_limit = 16384
        self._privileged_cache_limit = 4096
        self._compiled_program = self._compile_program()
        self._weight_terms = self._build_weight_terms()

    def _default_graded_safety_weights(self) -> dict[str, dict[str, float]]:
        return {
            "must_stop": {"slow": 0.05, "normal": 0.01, "fast": 0.0, "stop": 1.0},
            "warning": {"slow": 0.90, "normal": 0.60, "fast": 0.10, "stop": 0.85},
            "fast_zone": {"slow": 0.55, "normal": 0.80, "fast": 1.00, "stop": 0.25},
            "normal_zone": {"slow": 0.75, "normal": 1.00, "fast": 0.35, "stop": 0.40},
        }

    def _resolve_action_names(self, action_space: list[str] | tuple[str, ...] | None) -> dict[int, str]:
        if action_space is None:
            action_labels = ["slow", "normal", "fast", "stop"]
        else:
            action_labels = [str(name).strip().lower() for name in action_space]
        if len(action_labels) != self.num_actions:
            raise ValueError(f"Expected {self.num_actions} action labels, got {len(action_labels)}.")
        if self.stop_action < 0 or self.stop_action >= self.num_actions:
            raise ValueError(f"stop_action index {self.stop_action} is out of range for {self.num_actions} actions.")
        if action_labels[self.stop_action] != "stop":
            raise ValueError(
                f"Configured stop_action={self.stop_action} must map to 'stop', got '{action_labels[self.stop_action]}'."
            )
        return {idx: label for idx, label in enumerate(action_labels)}

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

    def _is_minimal_observation(self) -> bool:
        minimal_keys = {"ego_in_inter", "other_in_inter", "ego_at_inter", "close_ahead", "ahead", "higher_pri"}
        return minimal_keys.issubset(set(self.pred_grounding_index.keys()))

    def _infer_num_entities(self) -> int:
        if self.minimal_observation:
            return 1
        for pred_name in ("IsCar", "IsPedestrian", "IsAtInter", "IsInInter"):
            if pred_name in self.pred_grounding_index:
                start, end = self.pred_grounding_index[pred_name]
                return int(end - start)
        if "HigherPri" in self.pred_grounding_index:
            start, end = self.pred_grounding_index["HigherPri"]
            width = int(round(np.sqrt(end - start)))
            return width
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

    def _unary_prob(self, obs: np.ndarray, pred_name: str, entity_idx: int) -> float:
        start, _ = self.pred_grounding_index[pred_name]
        return float(np.clip(obs[start + entity_idx], 0.0, 1.0))

    def _scalar_prob(self, obs: np.ndarray, pred_name: str) -> float:
        start, _ = self.pred_grounding_index[pred_name]
        return float(np.clip(obs[start], 0.0, 1.0))

    def _binary_prob(self, obs: np.ndarray, pred_name: str, entity_i: int, entity_j: int) -> float:
        start, _ = self.pred_grounding_index[pred_name]
        offset = entity_i * self.n_entities + entity_j
        return float(np.clip(obs[start + offset], 0.0, 1.0))

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

    def _prefer_safe_motion(self, safe_probs: np.ndarray) -> np.ndarray:
        adjusted = np.asarray(safe_probs, dtype=np.float32).copy()
        if self.stop_priority_scale >= 1.0:
            return adjusted
        move_indices = [idx for idx in range(self.num_actions) if idx != self.stop_action]
        if not move_indices:
            return adjusted
        max_move = float(np.max(adjusted[move_indices]))
        if max_move <= 0.0:
            return adjusted
        adjusted[self.stop_action] = min(float(adjusted[self.stop_action]), float(self.stop_priority_scale * max_move))
        return adjusted

    def register_privileged_context(self, obs: Any, context: dict[str, Any] | None) -> None:
        if (not self.use_privileged_internal_safety) or context is None:
            return
        joint_context = context.get("joint_intersection_context")
        if not isinstance(joint_context, dict):
            return
        if self.minimal_observation:
            normalized = {
                "other_in_inter": 0.0,
                "higher_pri": 0.0,
            }
            other_in = joint_context.get("other_in_inter", []) or []
            other_at = joint_context.get("other_at_inter", []) or []
            other_is_car = joint_context.get("other_is_car", []) or []
            higher_pri = joint_context.get("higher_pri", []) or []
            n_other = max(len(other_in), len(other_at), len(other_is_car), len(higher_pri))
            for idx in range(n_other):
                is_car = float(other_is_car[idx]) if idx < len(other_is_car) else 0.0
                in_inter = float(other_in[idx]) if idx < len(other_in) else 0.0
                at_inter = float(other_at[idx]) if idx < len(other_at) else 0.0
                has_higher_pri = float(higher_pri[idx]) if idx < len(higher_pri) else 0.0
                if in_inter > 0.0:
                    normalized["other_in_inter"] = 1.0
                if max(at_inter, in_inter) > 0.0 and has_higher_pri > 0.0:
                    normalized["higher_pri"] = 1.0
        else:
            normalized = {
                "ego_at_inter": float(np.clip(joint_context.get("ego_at_inter", 0.0), 0.0, 1.0)),
                "ego_in_inter": float(np.clip(joint_context.get("ego_in_inter", 0.0), 0.0, 1.0)),
                "other_at_inter": [float(np.clip(v, 0.0, 1.0)) for v in joint_context.get("other_at_inter", [])],
                "other_in_inter": [float(np.clip(v, 0.0, 1.0)) for v in joint_context.get("other_in_inter", [])],
                "other_is_car": [float(np.clip(v, 0.0, 1.0)) for v in joint_context.get("other_is_car", [])],
                "higher_pri": [float(np.clip(v, 0.0, 1.0)) for v in joint_context.get("higher_pri", [])],
            }
        obs_key = self._obs_cache_key(obs)
        self._cache_put(self._privileged_context_cache, obs_key, normalized, self._privileged_cache_limit)

    def _context_signature(self, context: dict[str, Any] | None) -> tuple[float, ...]:
        if not context:
            return ()
        if self.minimal_observation:
            return (
                float(context.get("other_in_inter", 0.0)),
                float(context.get("higher_pri", 0.0)),
            )
        signature = [
            float(context.get("ego_at_inter", 0.0)),
            float(context.get("ego_in_inter", 0.0)),
        ]
        for key in ("other_at_inter", "other_in_inter", "higher_pri"):
            values = context.get(key, [])
            signature.extend(float(v) for v in values[: self.n_entities - 1])
        return tuple(signature)

    def _state_probabilities(self, obs: Any) -> dict[str, float]:
        obs_key = self._obs_cache_key(obs)
        privileged_context = None
        if self.use_privileged_internal_safety:
            privileged_context = self._cache_get(self._privileged_context_cache, obs_key)
        cache_key = ("ctx",) + obs_key + self._context_signature(privileged_context)
        cached = self._cache_get(self._state_prob_cache, cache_key)
        if cached is not None:
            return cached

        obs_vec = self._prepare_obs(obs)
        if self.minimal_observation:
            probs = {
                "ego_in_inter": self._scalar_prob(obs_vec, "ego_in_inter"),
                "other_in_inter": self._scalar_prob(obs_vec, "other_in_inter"),
                "ego_at_inter": self._scalar_prob(obs_vec, "ego_at_inter"),
                "close_ahead": self._scalar_prob(obs_vec, "close_ahead"),
                "ahead": self._scalar_prob(obs_vec, "ahead"),
                "higher_pri": self._scalar_prob(obs_vec, "higher_pri"),
            }
            if privileged_context is not None:
                if "other_in_inter" in privileged_context:
                    probs["other_in_inter"] = float(np.clip(privileged_context["other_in_inter"], 0.0, 1.0))
                if "higher_pri" in privileged_context:
                    probs["higher_pri"] = float(np.clip(privileged_context["higher_pri"], 0.0, 1.0))
        else:
            probs = {
                "ego_at_inter": self._unary_prob(obs_vec, "IsAtInter", 0),
                "ego_in_inter": self._unary_prob(obs_vec, "IsInInter", 0),
            }
            for other_idx in range(1, self.n_entities):
                probs[f"higher_pri_{other_idx}"] = self._binary_prob(obs_vec, "HigherPri", other_idx, 0)
                probs[f"other_in_inter_{other_idx}"] = self._unary_prob(obs_vec, "IsInInter", other_idx)
                probs[f"other_at_inter_{other_idx}"] = self._unary_prob(obs_vec, "IsAtInter", other_idx)
                probs[f"other_is_car_{other_idx}"] = self._unary_prob(obs_vec, "IsCar", other_idx) if "IsCar" in self.pred_grounding_index else 0.0
                probs[f"ahead_{other_idx}"] = self._binary_prob(obs_vec, "IsAhead", 0, other_idx) if "IsAhead" in self.pred_grounding_index else 0.0
                probs[f"close_ahead_{other_idx}"] = self._binary_prob(obs_vec, "IsCloseAhead", 0, other_idx) if "IsCloseAhead" in self.pred_grounding_index else 0.0

            if privileged_context is not None:
                for other_idx in range(1, self.n_entities):
                    ctx_idx = other_idx - 1
                    other_at = privileged_context.get("other_at_inter", [])
                    other_in = privileged_context.get("other_in_inter", [])
                    other_is_car = privileged_context.get("other_is_car", [])
                    higher = privileged_context.get("higher_pri", [])
                    if ctx_idx < len(other_at):
                        probs[f"other_at_inter_{other_idx}"] = float(np.clip(other_at[ctx_idx], 0.0, 1.0))
                    if ctx_idx < len(other_in):
                        probs[f"other_in_inter_{other_idx}"] = float(np.clip(other_in[ctx_idx], 0.0, 1.0))
                    if ctx_idx < len(other_is_car):
                        probs[f"other_is_car_{other_idx}"] = float(np.clip(other_is_car[ctx_idx], 0.0, 1.0))
                    if ctx_idx < len(higher):
                        probs[f"higher_pri_{other_idx}"] = float(np.clip(higher[ctx_idx], 0.0, 1.0))

        self._cache_put(self._state_prob_cache, cache_key, probs, self._state_cache_limit)
        return probs

    def _local_state_probabilities(self, obs: Any) -> dict[str, float]:
        obs_vec = self._prepare_obs(obs)
        if self.minimal_observation:
            return {
                "ego_in_inter": self._scalar_prob(obs_vec, "ego_in_inter"),
                "other_in_inter": self._scalar_prob(obs_vec, "other_in_inter"),
                "ego_at_inter": self._scalar_prob(obs_vec, "ego_at_inter"),
                "close_ahead": self._scalar_prob(obs_vec, "close_ahead"),
                "ahead": self._scalar_prob(obs_vec, "ahead"),
                "higher_pri": self._scalar_prob(obs_vec, "higher_pri"),
            }
        probs = {
            "ego_at_inter": self._unary_prob(obs_vec, "IsAtInter", 0),
            "ego_in_inter": self._unary_prob(obs_vec, "IsInInter", 0),
        }
        for other_idx in range(1, self.n_entities):
            probs[f"higher_pri_{other_idx}"] = self._binary_prob(obs_vec, "HigherPri", other_idx, 0)
            probs[f"other_in_inter_{other_idx}"] = self._unary_prob(obs_vec, "IsInInter", other_idx)
            probs[f"other_at_inter_{other_idx}"] = self._unary_prob(obs_vec, "IsAtInter", other_idx)
            probs[f"other_is_car_{other_idx}"] = self._unary_prob(obs_vec, "IsCar", other_idx) if "IsCar" in self.pred_grounding_index else 0.0
            probs[f"ahead_{other_idx}"] = self._binary_prob(obs_vec, "IsAhead", 0, other_idx) if "IsAhead" in self.pred_grounding_index else 0.0
            probs[f"close_ahead_{other_idx}"] = self._binary_prob(obs_vec, "IsCloseAhead", 0, other_idx) if "IsCloseAhead" in self.pred_grounding_index else 0.0
        return probs

    def debug_snapshot(self, obs: Any) -> dict[str, Any]:
        obs_np = self._prepare_obs(obs)
        local_facts = self._local_state_probabilities(obs_np)
        final_facts = self._state_probabilities(obs_np)
        obs_key = self._obs_cache_key(obs_np)
        privileged_context = self._cache_get(self._privileged_context_cache, obs_key) if self.use_privileged_internal_safety else None
        overridden = {}
        for key, final_value in final_facts.items():
            local_value = local_facts.get(key)
            if local_value is None:
                continue
            if abs(float(final_value) - float(local_value)) > 1e-6:
                overridden[key] = {"local": float(local_value), "final": float(final_value)}
        return {
            "local_facts": local_facts,
            "final_facts": final_facts,
            "privileged_context": privileged_context,
            "overridden_facts": overridden,
        }

    def _program_atom_names(self) -> list[str]:
        if self.minimal_observation:
            return ["ego_in_inter", "other_in_inter", "ego_at_inter", "close_ahead", "ahead", "higher_pri"]
        atom_names = ["ego_at_inter", "ego_in_inter"]
        for other_idx in range(1, self.n_entities):
            atom_names.extend(
                [
                    f"higher_pri_{other_idx}",
                    f"other_in_inter_{other_idx}",
                    f"other_at_inter_{other_idx}",
                    f"other_is_car_{other_idx}",
                    f"ahead_{other_idx}",
                    f"close_ahead_{other_idx}",
                ]
            )
        return atom_names

    def _format_prob(self, probability: float) -> str:
        return f"{float(np.clip(probability, 0.0, 1.0)):.10f}"

    def _build_program_template(self) -> str:
        lines = []
        action_labels = [self.action_names[idx] for idx in range(self.num_actions)]
        action_heads = "; ".join(f"0.25::act({label})" for label in action_labels)
        lines.append(f"{action_heads}.")
        for atom_name in self._program_atom_names():
            lines.append(f"0.5::{atom_name}.")

        if self.minimal_observation:
            lines.extend(
                [
                    "must_stop :- close_ahead.",
                    "must_stop :- ego_at_inter, other_in_inter.",
                    "must_stop :- ego_at_inter, higher_pri.",
                    "must_stop :- ego_in_inter, other_in_inter, higher_pri.",
                    "warning_state :- \\+ must_stop, ahead.",
                    "warning_state :- \\+ must_stop, ego_at_inter.",
                    "warning_state :- \\+ must_stop, ego_in_inter, other_in_inter.",
                    "fast_zone_state :- \\+ must_stop, \\+ warning_state, \\+ ego_at_inter, \\+ ego_in_inter.",
                    "normal_zone_state :- \\+ must_stop, \\+ warning_state, \\+ fast_zone_state.",
                ]
            )
        else:
            for other_idx in range(1, self.n_entities):
                lines.append(f"occupancy_conflict_{other_idx} :- ego_at_inter, other_in_inter_{other_idx}.")
                lines.append(f"waiting_conflict_{other_idx} :- ego_at_inter, other_is_car_{other_idx}, higher_pri_{other_idx}, other_at_inter_{other_idx}.")
                lines.append(f"in_intersection_priority_conflict_{other_idx} :- ego_in_inter, other_is_car_{other_idx}, higher_pri_{other_idx}, other_in_inter_{other_idx}.")
                lines.append(f"colliding_conflict_{other_idx} :- close_ahead_{other_idx}.")
                lines.append(f"ahead_warning_{other_idx} :- ahead_{other_idx}.")
                lines.append(f"must_stop :- occupancy_conflict_{other_idx}.")
                lines.append(f"must_stop :- waiting_conflict_{other_idx}.")
                lines.append(f"must_stop :- in_intersection_priority_conflict_{other_idx}.")
                lines.append(f"must_stop :- colliding_conflict_{other_idx}.")
                lines.append(f"has_ahead_warning :- ahead_warning_{other_idx}.")

            lines.extend(
                [
                    "warning_state :- \\+ must_stop, has_ahead_warning.",
                    "fast_zone_state :- \\+ must_stop, \\+ has_ahead_warning, \\+ ego_at_inter, \\+ ego_in_inter.",
                    "normal_zone_state :- \\+ must_stop, \\+ has_ahead_warning, \\+ fast_zone_state.",
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
                prob = self._format_prob(self.graded_safety_weights[regime_name][action_name])
                fact_name = f"graded_{regime_name}_{action_name}"
                lines.append(f"{prob}::{fact_name}.")
                lines.append(f"safe_action_{action_name} :- {state_atom}, {fact_name}.")

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
        return get_evaluatable().create_from(PrologString(self._build_program_template()))

    def _build_weight_terms(self) -> dict[str, Any]:
        weight_terms = {atom_name: Term(atom_name) for atom_name in self._program_atom_names()}
        for action_name in self.action_names.values():
            weight_terms[f"act_{action_name}"] = Term("act", Term(action_name))
        return weight_terms

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
            weights[self._weight_terms[term_key]] = float(np.clip(policy_probs[idx], 0.0, 1.0))

        result = self._compiled_program.evaluate(weights=weights)
        parsed = {str(key): float(value) for key, value in result.items()}
        self._cache_put(self._compiled_eval_cache, cache_key, parsed, self._compiled_cache_limit)
        return parsed

    def _result_for_obs_policy(self, obs: Any, policy_probs: Any | None) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
        probs = np.ones(self.num_actions, dtype=np.float32) / float(self.num_actions) if policy_probs is None else self._policy_probs(policy_probs)
        state_probs = self._state_probabilities(obs)
        result = self._evaluate_program(state_probs, probs)
        return probs, state_probs, result

    def _safe_action_probs_from_result(self, result: dict[str, float], policy_probs: np.ndarray) -> np.ndarray:
        joint = np.array(
            [
                float(result.get(f"joint_safe_{self.action_names[idx]}", 0.0))
                for idx in range(self.num_actions)
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
        safe = self._prefer_safe_motion(self._safe_action_probs_from_result(result, probs))
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
