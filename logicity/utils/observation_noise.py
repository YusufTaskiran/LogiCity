import math
from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np


TYPE_PREDICATES = {
    "IsCar",
    "IsBus",
    "IsPedestrian",
    "IsAmbulance",
    "IsPolice",
    "IsOld",
    "IsReckless",
    "IsTiro",
}


def _apply_asymmetric_binary_noise(values: np.ndarray, false_negative: float, false_positive: float) -> np.ndarray:
    noisy = values.copy()
    ones = noisy > 0.5
    if false_negative > 0.0:
        flip_to_zero = np.random.rand(*noisy.shape) < false_negative
        noisy[np.logical_and(ones, flip_to_zero)] = 0.0
    if false_positive > 0.0:
        flip_to_one = np.random.rand(*noisy.shape) < false_positive
        noisy[np.logical_and(~ones, flip_to_one)] = 1.0
    return noisy


def _infer_num_entities(pred_grounding_index: Optional[Dict[str, Tuple[int, int]]]) -> Optional[int]:
    if not pred_grounding_index:
        return None
    if "IsAtInter" in pred_grounding_index:
        start, end = pred_grounding_index["IsAtInter"]
        return end - start
    for _pred_name, (start, end) in pred_grounding_index.items():
        length = end - start
        root = int(round(math.sqrt(length)))
        if root > 1 and root * root == length:
            return root
    return None


def _entity_groups(pred_grounding_index: Dict[str, Tuple[int, int]]) -> Optional[Dict[int, Dict[str, Set[int]]]]:
    num_entities = _infer_num_entities(pred_grounding_index)
    if num_entities is None or num_entities <= 1:
        return None

    groups: Dict[int, Dict[str, Set[int]]] = {
        entity_id: {"type": set(), "relation": set()} for entity_id in range(num_entities)
    }

    for pred_name, (start, end) in pred_grounding_index.items():
        length = end - start
        if length == num_entities:
            bucket = "type" if pred_name in TYPE_PREDICATES else "relation"
            for entity_id in range(num_entities):
                groups[entity_id][bucket].add(start + entity_id)
            continue

        root = int(round(math.sqrt(length)))
        if root > 1 and root == num_entities and root * root == length:
            for row in range(num_entities):
                for col in range(num_entities):
                    idx = start + row * num_entities + col
                    groups[row]["relation"].add(idx)
                    groups[col]["relation"].add(idx)

    return groups


def apply_observation_noise(
    obs_array: np.ndarray,
    observation_noise: Optional[Dict],
    pred_grounding_index: Optional[Dict[str, Tuple[int, int]]] = None,
) -> np.ndarray:
    obs = np.asarray(obs_array, dtype=np.float32).copy()
    if not observation_noise:
        return obs

    mode = observation_noise.get("mode")
    if not mode:
        return obs

    epsilon = float(observation_noise.get("epsilon", 0.0))
    epsilon = min(max(epsilon, 0.0), 0.5)

    if mode == "soft_symmetric":
        if epsilon <= 0.0:
            return obs
        ones = obs > 0.5
        obs[ones] = 1.0 - epsilon
        obs[~ones] = epsilon
        return obs

    if mode == "bit_flip":
        if epsilon <= 0.0:
            return obs
        flips = np.random.rand(*obs.shape) < epsilon
        binary = (obs > 0.5).astype(np.float32)
        binary[flips] = 1.0 - binary[flips]
        return binary

    if mode != "entity_correlated_asymmetric":
        return obs

    if pred_grounding_index is None:
        return obs

    groups = _entity_groups(pred_grounding_index)
    if not groups:
        return obs

    base_binary = (obs > 0.5).astype(np.float32)
    noisy = base_binary.copy()
    entity_dropout = float(observation_noise.get("entity_dropout", epsilon))
    entity_dropout = min(max(entity_dropout, 0.0), 1.0)
    output_mode = observation_noise.get("output_mode", "soft_probability")
    sample_binary = output_mode == "sampled_binary"

    false_negative_type = float(observation_noise.get("false_negative_type", max(0.01, epsilon * 0.4)))
    false_positive_type = float(observation_noise.get("false_positive_type", max(0.005, epsilon * 0.1)))
    false_negative_relation = float(observation_noise.get("false_negative_relation", max(0.02, epsilon * 0.8)))
    false_positive_relation = float(observation_noise.get("false_positive_relation", max(0.005, epsilon * 0.15)))
    occluded_false_negative_relation = float(
        observation_noise.get("occluded_false_negative_relation", min(0.5, false_negative_relation * 2.0))
    )
    occluded_false_positive_relation = float(
        observation_noise.get("occluded_false_positive_relation", min(0.25, false_positive_relation * 2.0))
    )
    ego_false_negative = float(observation_noise.get("ego_false_negative", max(0.0, epsilon * 0.1)))
    ego_false_positive = float(observation_noise.get("ego_false_positive", 0.0))

    def to_confidence(values: np.ndarray, false_negative: float, false_positive: float) -> np.ndarray:
        ones = values > 0.5
        confidence = np.empty_like(values, dtype=np.float32)
        confidence[ones] = 1.0 - false_negative
        confidence[~ones] = false_positive
        return np.clip(confidence, 0.0, 1.0)

    def maybe_sample(confidence: np.ndarray) -> np.ndarray:
        if not sample_binary:
            return confidence
        draws = np.random.rand(*confidence.shape)
        return (draws < confidence).astype(np.float32)

    for entity_id, entity_group in groups.items():
        type_indices = sorted(entity_group["type"])
        relation_indices = sorted(entity_group["relation"])

        if entity_id == 0:
            if type_indices:
                confidence = to_confidence(base_binary[type_indices], ego_false_negative, ego_false_positive)
                noisy[type_indices] = maybe_sample(confidence)
            if relation_indices:
                confidence = to_confidence(base_binary[relation_indices], ego_false_negative, ego_false_positive)
                noisy[relation_indices] = maybe_sample(confidence)
            continue

        occluded = np.random.rand() < entity_dropout
        rel_fn = occluded_false_negative_relation if occluded else false_negative_relation
        rel_fp = occluded_false_positive_relation if occluded else false_positive_relation

        if type_indices:
            confidence = to_confidence(base_binary[type_indices], false_negative_type, false_positive_type)
            noisy[type_indices] = maybe_sample(confidence)
        if relation_indices:
            confidence = to_confidence(base_binary[relation_indices], rel_fn, rel_fp)
            noisy[relation_indices] = maybe_sample(confidence)

    return noisy.astype(np.float32)
