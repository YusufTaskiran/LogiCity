from __future__ import annotations

from typing import Any

import numpy as np


class PredicateSensorModel:
    """Simple confusion-matrix sensor model for shield predicates.

    The simulator still provides ground-truth binary predicates. This model
    perturbs only the shield-side perception for selected predicates.

    For a true predicate z in {0, 1}:
    - if z = 1, the sensor reports positive with probability tpr
    - if z = 0, the sensor reports positive with probability fpr

    The sensor then exposes a soft confidence value:
    - positive observation -> positive_confidence
    - negative observation -> negative_confidence
    """

    DEFAULT_ACTION_PROFILES = {
        "slow": {"tpr": 0.97, "fpr": 0.03},
        "normal": {"tpr": 0.94, "fpr": 0.06},
        "fast": {"tpr": 0.90, "fpr": 0.10},
        "stop": {"tpr": 0.99, "fpr": 0.01},
    }

    DEFAULT_PREDICATES = {
        "IsAtInter": {
            "slow": {"tpr": 0.97, "fpr": 0.03},
            "normal": {"tpr": 0.94, "fpr": 0.06},
            "fast": {"tpr": 0.90, "fpr": 0.10},
            "stop": {"tpr": 0.99, "fpr": 0.01},
        },
        "IsInInter": {
            "slow": {"tpr": 0.97, "fpr": 0.03},
            "normal": {"tpr": 0.94, "fpr": 0.06},
            "fast": {"tpr": 0.90, "fpr": 0.10},
            "stop": {"tpr": 0.99, "fpr": 0.01},
        },
        "CollidingClose": {
            "slow": {"tpr": 0.92, "fpr": 0.08},
            "normal": {"tpr": 0.88, "fpr": 0.12},
            "fast": {"tpr": 0.82, "fpr": 0.18},
            "stop": {"tpr": 0.97, "fpr": 0.03},
        },
    }

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.enabled = bool(config.get("enabled", False))
        self.mode = str(config.get("mode", "confusion_matrix"))
        self.threshold = float(config.get("threshold", 0.5))
        self.positive_confidence = float(config.get("positive_confidence", 0.9))
        self.negative_confidence = float(config.get("negative_confidence", 0.1))
        predicates_cfg = config.get("predicates", {})
        self.predicates = {}
        for name, default_profiles in self.DEFAULT_PREDICATES.items():
            configured_profiles = predicates_cfg.get(name, {})
            merged_profiles = {}
            for action_name, action_defaults in default_profiles.items():
                override = configured_profiles.get(action_name, {})
                merged_profiles[action_name] = {
                    "tpr": float(override.get("tpr", action_defaults["tpr"])),
                    "fpr": float(override.get("fpr", action_defaults["fpr"])),
                }
            self.predicates[name] = merged_profiles

    def is_uncertain(self, pred_name: str) -> bool:
        return self.enabled and pred_name in self.predicates

    def sense_probability(self, pred_name: str, truth_value: bool, action_name: str | None = None) -> float:
        truth_value = bool(truth_value)
        if (not self.enabled) or pred_name not in self.predicates:
            return 1.0 if truth_value else 0.0
        if self.mode != "confusion_matrix":
            raise ValueError(f"Unsupported sensor uncertainty mode: {self.mode}")
        action_name = action_name or "normal"
        pred_cfg = self.predicates[pred_name].get(action_name)
        if pred_cfg is None:
            pred_cfg = self.DEFAULT_ACTION_PROFILES.get(action_name, self.DEFAULT_ACTION_PROFILES["normal"])
        positive_prob = pred_cfg["tpr"] if truth_value else pred_cfg["fpr"]
        observed_positive = bool(np.random.random() < positive_prob)
        return self.positive_confidence if observed_positive else self.negative_confidence

    def threshold_probability(self, probability: float) -> bool:
        return bool(float(probability) >= self.threshold)
