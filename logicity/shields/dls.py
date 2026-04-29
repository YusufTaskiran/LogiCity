import math
from typing import Any

import numpy as np


class DeterministicLogicShield:
    """Deterministic probability-level logic shield.

    In the current thesis setup the discrete action space is:
    0 = slow, 1 = normal, 2 = fast, 3 = stop.

    The shield computes a hard safety mask P(safe | s, a) in {0, 1} and
    renormalizes the base policy over safe actions only.

    In the current thesis setup the deterministic regime structure is:

    - must_stop -> only `stop` safe
    - warning -> `slow` and `stop` safe
    - fast_zone -> `slow`, `normal`, `fast`, `stop` safe
    - normal_zone -> `slow`, `normal`, `stop` safe
    """

    def __init__(
        self,
        pred_grounding_index: dict[str, tuple[int, int]],
        num_actions: int = 4,
        safe_action: int = 3,
    ):
        self.pred_grounding_index = pred_grounding_index
        self.num_actions = int(num_actions)
        self.safe_action = int(safe_action)
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
        return bool(observed_value > 0.5)

    def _binary(self, obs: np.ndarray, pred_name: str, entity_i: int, entity_j: int, action_name: str) -> bool:
        start, _ = self.pred_grounding_index[pred_name]
        offset = entity_i * self.n_entities + entity_j
        observed_value = float(obs[start + offset])
        return bool(observed_value > 0.5)

    def _state_facts(self, obs: Any) -> dict[str, Any]:
        obs_vec = self._prepare_obs(obs)
        ego_at_inter = self._unary(obs_vec, "IsAtInter", 0, "ego")
        ego_in_inter = self._unary(obs_vec, "IsInInter", 0, "ego")
        higher_pri = []
        other_in_inter = []
        other_at_inter = []
        colliding_close = []
        close_ahead = []
        for other_idx in range(1, self.n_entities):
            higher_pri.append(self._binary(obs_vec, "HigherPri", other_idx, 0, "other"))
            other_in_inter.append(self._unary(obs_vec, "IsInInter", other_idx, "other"))
            other_at_inter.append(self._unary(obs_vec, "IsAtInter", other_idx, "other"))
            colliding_close.append(self._binary(obs_vec, "CollidingClose", 0, other_idx, "other"))
            close_ahead.append(self._binary(obs_vec, "IsCloseAhead", 0, other_idx, "other"))
        return {
            "ego_at_inter": ego_at_inter,
            "ego_in_inter": ego_in_inter,
            "higher_pri": higher_pri,
            "other_in_inter": other_in_inter,
            "other_at_inter": other_at_inter,
            "colliding_close": colliding_close,
            "close_ahead": close_ahead,
        }

    def _must_stop(self, facts: dict[str, Any]) -> bool:
        ego_at_inter = bool(facts["ego_at_inter"])
        for idx in range(len(facts["higher_pri"])):
            if ego_at_inter and bool(facts["other_in_inter"][idx]):
                return True
            if ego_at_inter and bool(facts["higher_pri"][idx]) and bool(facts["other_at_inter"][idx]):
                return True
            if bool(facts["colliding_close"][idx]):
                return True
        return False

    def _has_close_ahead(self, facts: dict[str, Any]) -> bool:
        return any(bool(v) for v in facts["close_ahead"])

    def _fast_zone(self, facts: dict[str, Any], must_stop: bool, has_close_ahead: bool) -> bool:
        return (not must_stop) and (not has_close_ahead) and (not bool(facts["ego_at_inter"])) and (not bool(facts["ego_in_inter"]))

    def stop_required(self, obs: Any, action_name: str) -> bool:
        facts = self._state_facts(obs)
        return self._must_stop(facts)

    def safety_probs(self, obs: Any) -> np.ndarray:
        self.total_calls += 1
        facts = self._state_facts(obs)
        must_stop = self._must_stop(facts)
        has_close_ahead = self._has_close_ahead(facts)
        fast_zone = self._fast_zone(facts, must_stop, has_close_ahead)

        slow_idx = 0
        normal_idx = 1 if self.num_actions > 1 else 0
        fast_idx = 2 if self.num_actions > 2 else normal_idx

        if must_stop:
            safe_probs = np.zeros(self.num_actions, dtype=np.float32)
            safe_probs[self.safe_action] = 1.0
        elif has_close_ahead:
            safe_probs = np.zeros(self.num_actions, dtype=np.float32)
            safe_probs[slow_idx] = 1.0
            safe_probs[self.safe_action] = 1.0
        elif fast_zone:
            safe_probs = np.ones(self.num_actions, dtype=np.float32)
        else:
            safe_probs = np.zeros(self.num_actions, dtype=np.float32)
            safe_probs[slow_idx] = 1.0
            safe_probs[normal_idx] = 1.0
            safe_probs[self.safe_action] = 1.0

        masked_any = bool(np.any(safe_probs[:-1] < 1.0))
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
