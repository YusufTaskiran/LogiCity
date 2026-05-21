from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

try:
    from problog import get_evaluatable
    from problog.logic import Term
    from problog.program import PrologString
except ImportError:  # pragma: no cover
    PrologString = None
    get_evaluatable = None
    Term = None


class JointProbabilisticLogicShieldTwoCar:
    """Minimal ProbLog joint shield for the two-car thesis edge cases.

    Joint facts used:
    - both_at_inter
    - same_priority
    - internal deterministic tie-breaker (`tie_break_a`)
    """

    def __init__(
        self,
        num_actions: int = 4,
        stop_action: int = 3,
        action_space: list[str] | tuple[str, ...] | None = None,
        graded_safety_weights: dict[str, dict[str, float]] | None = None,
    ):
        if PrologString is None or get_evaluatable is None or Term is None:
            raise ImportError("ProbLog backend requested, but the `problog` package is not installed.")
        self.num_actions = int(num_actions)
        self.stop_action = int(stop_action)
        self.action_names = self._resolve_action_names(action_space)
        self.graded_safety_weights = self._normalize_graded_safety_weights(graded_safety_weights)
        self.intervention_count = 0
        self.total_calls = 0
        self._compiled_program = self._compile_program()
        self._weight_terms = self._build_weight_terms()
        self._eval_cache: OrderedDict[tuple[tuple[str, float], ...], dict[str, float]] = OrderedDict()
        self._cache_limit = 4096

    def _resolve_action_names(self, action_space: list[str] | tuple[str, ...] | None) -> dict[int, str]:
        if action_space is None:
            action_labels = ["slow", "normal", "fast", "stop"]
        else:
            action_labels = [str(name).strip().lower() for name in action_space]
        if len(action_labels) != self.num_actions:
            raise ValueError(f"Expected {self.num_actions} action labels, got {len(action_labels)}.")
        return {idx: label for idx, label in enumerate(action_labels)}

    def _default_graded_safety_weights(self) -> dict[str, dict[str, float]]:
        return {
            "must_stop": {"slow": 0.05, "normal": 0.01, "fast": 0.0, "stop": 1.0},
            "warning": {"slow": 0.90, "normal": 0.60, "fast": 0.10, "stop": 0.85},
            "fast_zone": {"slow": 0.55, "normal": 0.80, "fast": 1.00, "stop": 0.25},
            "normal_zone": {"slow": 0.75, "normal": 1.00, "fast": 0.35, "stop": 0.40},
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

    def reset_metrics(self) -> None:
        self.intervention_count = 0
        self.total_calls = 0

    def get_metrics(self) -> dict[str, float]:
        rate = float(self.intervention_count / self.total_calls) if self.total_calls > 0 else 0.0
        return {
            "joint_shield_intervention_count": int(self.intervention_count),
            "joint_shield_intervention_rate": rate,
        }

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

    def _build_program_template(self) -> str:
        labels = [self.action_names[idx] for idx in range(self.num_actions)]
        lines = [
            "0.5::both_at_inter.",
            "0.5::same_priority.",
            "0.5::tie_break_a.",
            "a_wins :- both_at_inter, same_priority, tie_break_a.",
            "b_wins :- both_at_inter, same_priority, \\+ tie_break_a.",
        ]
        for label in labels:
            lines.append(f"0.25::act_a_{label}.")
            lines.append(f"0.25::act_b_{label}.")

        for regime_name, action_weights in self.graded_safety_weights.items():
            for label in labels:
                prob = float(np.clip(action_weights[label], 0.0, 1.0))
                lines.append(f"{prob:.10f}::graded_{regime_name}_{label}_a.")
                lines.append(f"{prob:.10f}::graded_{regime_name}_{label}_b.")

        for a_label in labels:
            for b_label in labels:
                atom = f"joint_safe_{a_label}_{b_label}"
                # Default regime away from the targeted coordination edge case:
                # use the same normal-zone weights as decentralized PLS.
                lines.append(
                    f"{atom} :- \\+ both_at_inter, act_a_{a_label}, act_b_{b_label}, "
                    f"graded_normal_zone_{a_label}_a, graded_normal_zone_{b_label}_b."
                )
                # When both cars are at the entry and priorities are not equal,
                # allow exactly one mover and one yielder. Base policy mass decides which side.
                if a_label != "stop" and b_label == "stop":
                    lines.append(
                        f"{atom} :- both_at_inter, \\+ same_priority, act_a_{a_label}, act_b_{b_label}, "
                        f"graded_warning_{a_label}_a, graded_must_stop_{b_label}_b."
                    )
                    lines.append(
                        f"{atom} :- a_wins, act_a_{a_label}, act_b_{b_label}, "
                        f"graded_warning_{a_label}_a, graded_must_stop_{b_label}_b."
                    )
                if a_label == "stop" and b_label != "stop":
                    lines.append(
                        f"{atom} :- both_at_inter, \\+ same_priority, act_a_{a_label}, act_b_{b_label}, "
                        f"graded_must_stop_{a_label}_a, graded_warning_{b_label}_b."
                    )
                    lines.append(
                        f"{atom} :- b_wins, act_a_{a_label}, act_b_{b_label}, "
                        f"graded_must_stop_{a_label}_a, graded_warning_{b_label}_b."
                    )
                lines.append(f"query({atom}).")
        return "\n".join(lines)

    def _compile_program(self):
        return get_evaluatable().create_from(PrologString(self._build_program_template()))

    def _build_weight_terms(self) -> dict[str, Any]:
        weight_terms = {
            "both_at_inter": Term("both_at_inter"),
            "same_priority": Term("same_priority"),
            "tie_break_a": Term("tie_break_a"),
        }
        for label in self.action_names.values():
            weight_terms[f"act_a_{label}"] = Term(f"act_a_{label}")
            weight_terms[f"act_b_{label}"] = Term(f"act_b_{label}")
        return weight_terms

    def _policy_probs(self, base_probs: Any) -> np.ndarray:
        probs = np.asarray(base_probs, dtype=np.float64).reshape(-1).copy()
        probs = np.clip(probs, 0.0, None)
        total = float(np.sum(probs))
        if total <= 0.0:
            probs = np.zeros(self.num_actions, dtype=np.float64)
            probs[self.stop_action] = 1.0
            return probs
        probs /= total
        return probs

    def _evaluate_program(
        self,
        base_probs_a: np.ndarray,
        base_probs_b: np.ndarray,
        joint_facts: dict[str, Any],
        tiebreak_order: tuple[int, int],
    ) -> dict[str, float]:
        tie_break_a = bool(int(tiebreak_order[0]) <= int(tiebreak_order[1]))
        cache_entries = [
            ("both_at_inter", round(float(bool(joint_facts.get("both_at_inter", False))), 6)),
            ("same_priority", round(float(bool(joint_facts.get("same_priority", False))), 6)),
            ("tie_break_a", round(float(tie_break_a), 6)),
        ]
        for idx, label in self.action_names.items():
            cache_entries.append((f"act_a_{label}", round(float(base_probs_a[idx]), 6)))
            cache_entries.append((f"act_b_{label}", round(float(base_probs_b[idx]), 6)))
        cache_key = tuple(cache_entries)
        cached = self._cache_get(self._eval_cache, cache_key)
        if cached is not None:
            return cached

        weights = {
            self._weight_terms["both_at_inter"]: float(bool(joint_facts.get("both_at_inter", False))),
            self._weight_terms["same_priority"]: float(bool(joint_facts.get("same_priority", False))),
            self._weight_terms["tie_break_a"]: float(tie_break_a),
        }
        for idx, label in self.action_names.items():
            weights[self._weight_terms[f"act_a_{label}"]] = float(np.clip(base_probs_a[idx], 0.0, 1.0))
            weights[self._weight_terms[f"act_b_{label}"]] = float(np.clip(base_probs_b[idx], 0.0, 1.0))

        result = self._compiled_program.evaluate(weights=weights)
        parsed = {str(key): float(value) for key, value in result.items()}
        self._cache_put(self._eval_cache, cache_key, parsed, self._cache_limit)
        return parsed

    def joint_pair_distribution(
        self,
        base_probs_a: np.ndarray,
        base_probs_b: np.ndarray,
        candidate_infos: dict[tuple[int, int], dict[str, Any]] | None,
        joint_facts: dict[str, Any],
        tiebreak_order: tuple[int, int],
    ) -> np.ndarray:
        probs_a = self._policy_probs(base_probs_a)
        probs_b = self._policy_probs(base_probs_b)
        safe_scores = self.joint_safe_scores(
            candidate_infos=candidate_infos,
            joint_facts=joint_facts,
            tiebreak_order=tiebreak_order,
        )
        pair_dist = np.outer(probs_a, probs_b) * safe_scores

        total = float(np.sum(pair_dist))
        if total <= 0.0:
            fallback = np.zeros_like(pair_dist, dtype=np.float64)
            fallback[self.stop_action, self.stop_action] = 1.0
            return fallback
        return pair_dist / total

    def joint_safe_scores(
        self,
        candidate_infos: dict[tuple[int, int], dict[str, Any]] | None,
        joint_facts: dict[str, Any],
        tiebreak_order: tuple[int, int],
    ) -> np.ndarray:
        candidate_infos = candidate_infos or {}
        uniform = np.ones(self.num_actions, dtype=np.float64) / float(self.num_actions)
        result = self._evaluate_program(uniform, uniform, joint_facts, tiebreak_order)
        pair_prior = (1.0 / float(self.num_actions)) ** 2

        safe_scores = np.zeros((self.num_actions, self.num_actions), dtype=np.float64)
        for a_idx, a_label in self.action_names.items():
            for b_idx, b_label in self.action_names.items():
                info = candidate_infos.get((int(a_idx), int(b_idx)), {})
                if bool(info.get("any_fail", False)):
                    continue
                joint_prob = float(result.get(f"joint_safe_{a_label}_{b_label}", 0.0))
                safe_scores[a_idx, b_idx] = float(np.clip(joint_prob / max(pair_prior, 1e-8), 0.0, 1.0))
        return safe_scores

    def debug_snapshot(
        self,
        base_probs_a: np.ndarray,
        base_probs_b: np.ndarray,
        candidate_infos: dict[tuple[int, int], dict[str, Any]] | None,
        joint_facts: dict[str, Any],
        tiebreak_order: tuple[int, int],
    ) -> dict[str, Any]:
        candidate_infos = candidate_infos or {}
        probs_a = self._policy_probs(base_probs_a)
        probs_b = self._policy_probs(base_probs_b)
        raw_result = self._evaluate_program(probs_a, probs_b, joint_facts, tiebreak_order)
        safe_scores = self.joint_safe_scores(
            candidate_infos=candidate_infos,
            joint_facts=joint_facts,
            tiebreak_order=tiebreak_order,
        )
        pair_dist = self.joint_pair_distribution(
            base_probs_a=probs_a,
            base_probs_b=probs_b,
            candidate_infos=candidate_infos,
            joint_facts=joint_facts,
            tiebreak_order=tiebreak_order,
        )
        labels = [self.action_names[idx] for idx in range(self.num_actions)]
        table = []
        for a_idx, a_label in enumerate(labels):
            for b_idx, b_label in enumerate(labels):
                info = candidate_infos.get((a_idx, b_idx), {})
                table.append(
                    {
                        "pair": f"({a_label},{b_label})",
                        "a_idx": a_idx,
                        "b_idx": b_idx,
                        "admissible": not bool(info.get("any_fail", False)),
                        "safe_score": float(safe_scores[a_idx, b_idx]),
                        "pair_prob": float(pair_dist[a_idx, b_idx]),
                        "raw_joint_safe_prob": float(raw_result.get(f"joint_safe_{a_label}_{b_label}", 0.0)),
                        "candidate_info": info,
                    }
                )
        table.sort(key=lambda row: row["pair_prob"], reverse=True)
        return {
            "joint_facts": {
                "both_at_inter": bool(joint_facts.get("both_at_inter", False)),
                "same_priority": bool(joint_facts.get("same_priority", False)),
                "tie_break_a": bool(int(tiebreak_order[0]) <= int(tiebreak_order[1])),
                "tiebreak_order": tuple(int(v) for v in tiebreak_order),
            },
            "base_probs_a": probs_a.tolist(),
            "base_probs_b": probs_b.tolist(),
            "safe_scores": safe_scores.tolist(),
            "pair_distribution": pair_dist.tolist(),
            "top_pairs": table[: min(8, len(table))],
        }

    def select_joint_action(
        self,
        base_probs_a: np.ndarray,
        base_probs_b: np.ndarray,
        candidate_infos: dict[tuple[int, int], dict[str, Any]] | None,
        joint_facts: dict[str, Any],
        tiebreak_order: tuple[int, int],
        deterministic: bool = True,
    ) -> tuple[tuple[int, int], np.ndarray]:
        self.total_calls += 1
        pair_dist = self.joint_pair_distribution(
            base_probs_a=base_probs_a,
            base_probs_b=base_probs_b,
            candidate_infos=candidate_infos,
            joint_facts=joint_facts,
            tiebreak_order=tiebreak_order,
        )

        greedy_independent = (
            int(np.argmax(np.asarray(base_probs_a, dtype=np.float64))),
            int(np.argmax(np.asarray(base_probs_b, dtype=np.float64))),
        )
        greedy_joint = tuple(int(v) for v in np.unravel_index(int(np.argmax(pair_dist)), pair_dist.shape))
        if greedy_joint != greedy_independent:
            self.intervention_count += 1

        if deterministic:
            return greedy_joint, pair_dist

        flat = pair_dist.reshape(-1)
        sampled = int(np.random.choice(np.arange(flat.shape[0]), p=flat))
        return tuple(int(v) for v in np.unravel_index(sampled, pair_dist.shape)), pair_dist
