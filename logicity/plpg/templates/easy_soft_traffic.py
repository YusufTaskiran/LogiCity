import numpy as np
import torch as th

from .base import BasePLPGTemplate


class EasySoftTrafficPLPGTemplate(BasePLPGTemplate):
    name = "easy_soft_traffic"
    action_names = ["slow", "normal", "fast", "stop"]
    COLLIDING_CLOSE_DAMPING = 0.85
    TRANSITION_APPROACH_SCALE = 1.00
    CAUTION_TRANSITION_THRESHOLD = 0.12
    EDGE_TRANSITION_THRESHOLD = 0.28
    CRITICAL_TRANSITION_THRESHOLD = 0.52

    def __init__(self, pred_grounding_index, rule_yaml_file=None, sensor_noise=None):
        super().__init__(pred_grounding_index, rule_yaml_file=rule_yaml_file, sensor_noise=sensor_noise)
        self.num_entities = pred_grounding_index["IsAtInter"][1] - pred_grounding_index["IsAtInter"][0]
        self._is_at_inter = pred_grounding_index["IsAtInter"]
        self._is_in_inter = pred_grounding_index["IsInInter"]
        self._same_inter = pred_grounding_index["SameInter"]
        self._higher_pri = pred_grounding_index["HigherPri"]
        self._colliding_close = pred_grounding_index["CollidingClose"]
        self._is_car = pred_grounding_index["IsCar"]
        self._is_pedestrian = pred_grounding_index["IsPedestrian"]
        self._obs_logic_end = max(
            self._is_at_inter[1],
            self._is_in_inter[1],
            self._same_inter[1],
            self._higher_pri[1],
            self._colliding_close[1],
            self._is_car[1],
            self._is_pedestrian[1],
        )

    def _pair_index(self, i, j):
        return i * self.num_entities + j

    def build_program_lines(self):
        return [
            "0.25::act(slow); 0.25::act(normal); 0.25::act(fast); 0.25::act(stop).",
            "safe.",
            "query(safe).",
            "query(act(slow)).",
            "query(act(normal)).",
            "query(act(fast)).",
            "query(act(stop)).",
        ]

    def obs_to_fact_weights(self, obs_row: th.Tensor):
        obs_logic = obs_row[: self._obs_logic_end]
        fact_weights = {
            "ego_is_car": self._prob_fact(obs_logic[self._is_car[0]].item()),
            "ego_is_at_inter": self._prob_fact(obs_logic[self._is_at_inter[0]].item()),
            "ego_is_in_inter": self._prob_fact(obs_logic[self._is_in_inter[0]].item()),
        }
        for i in range(1, self.num_entities):
            fact_weights[f"other_{i}_is_at_inter"] = self._prob_fact(obs_logic[self._is_at_inter[0] + i].item())
            fact_weights[f"other_{i}_is_in_inter"] = self._prob_fact(obs_logic[self._is_in_inter[0] + i].item())
            fact_weights[f"other_{i}_same_inter"] = self._prob_fact(
                obs_logic[self._same_inter[0] + self._pair_index(0, i)].item()
            )
            fact_weights[f"other_{i}_higher_pri"] = self._prob_fact(
                obs_logic[self._higher_pri[0] + self._pair_index(i, 0)].item()
            )
            fact_weights[f"other_{i}_colliding_close"] = self._prob_fact(
                obs_logic[self._colliding_close[0] + self._pair_index(0, i)].item()
            )
            fact_weights[f"other_{i}_is_pedestrian"] = self._prob_fact(
                obs_logic[self._is_pedestrian[0] + i].item()
            )
        return fact_weights

    def _soft_rule_hazards(self, fact_weights):
        ego_is_car = float(fact_weights["ego_is_car"])
        ego_at = float(fact_weights["ego_is_at_inter"])
        ego_in = float(fact_weights["ego_is_in_inter"])

        colliding_close = 0.0
        ped_conflict = 0.0
        occupied_intersection = 0.0
        higher_priority = 0.0
        simultaneous_in_inter = 0.0
        transition_conflict = 0.0
        for i in range(1, self.num_entities):
            same_inter = float(fact_weights[f"other_{i}_same_inter"])
            other_at = float(fact_weights[f"other_{i}_is_at_inter"])
            other_in = float(fact_weights[f"other_{i}_is_in_inter"])
            other_ped = float(fact_weights[f"other_{i}_is_pedestrian"])
            other_higher = float(fact_weights[f"other_{i}_higher_pri"])
            other_colliding_close = float(fact_weights[f"other_{i}_colliding_close"])

            overlap_prob = max(
                ego_at * other_at,
                ego_at * other_in,
                ego_in * other_at,
                ego_in * other_in,
            )

            colliding_close = max(colliding_close, self.COLLIDING_CLOSE_DAMPING * other_colliding_close)
            ped_conflict = max(
                ped_conflict,
                ego_is_car * other_ped * same_inter * overlap_prob,
            )
            occupied_intersection = max(
                occupied_intersection,
                ego_at * same_inter * other_in * (1.0 - 0.5 * other_ped),
            )
            higher_priority = max(
                higher_priority,
                ego_at * other_at * same_inter * other_higher * (1.0 - other_ped),
            )
            simultaneous_in_inter = max(
                simultaneous_in_inter,
                ego_is_car * (1.0 - other_ped) * same_inter * ego_in * other_in,
            )
            occupancy_cue = max(other_in, other_at * (1.0 - other_ped))
            ped_transition = ego_is_car * other_ped * same_inter * max(
                other_in,
                0.70 * other_colliding_close,
                0.35 * other_at,
            )
            veh_transition = ego_is_car * (1.0 - other_ped) * same_inter * max(
                occupancy_cue,
                0.60 * other_colliding_close,
            )
            transition_conflict = max(
                transition_conflict,
                self.TRANSITION_APPROACH_SCALE * max(ped_transition, veh_transition),
            )

        soft_hazard = max(
            colliding_close,
            ped_conflict,
            occupied_intersection,
            higher_priority,
            simultaneous_in_inter,
            transition_conflict,
        )
        return {
            "colliding_close": float(np.clip(colliding_close, 0.0, 1.0)),
            "ped_conflict": float(np.clip(ped_conflict, 0.0, 1.0)),
            "occupied_intersection": float(np.clip(occupied_intersection, 0.0, 1.0)),
            "higher_priority": float(np.clip(higher_priority, 0.0, 1.0)),
            "simultaneous_in_inter": float(np.clip(simultaneous_in_inter, 0.0, 1.0)),
            "transition_conflict": float(np.clip(transition_conflict, 0.0, 1.0)),
            "hazard": float(np.clip(soft_hazard, 0.0, 1.0)),
        }

    def fact_weights_to_metrics(self, fact_weights):
        return self._soft_rule_hazards(fact_weights)

    def _regime_scores(self, hazards, fact_weights):
        ego_at = float(fact_weights["ego_is_at_inter"])
        ego_in = float(fact_weights["ego_is_in_inter"])
        colliding_close = hazards["colliding_close"]
        ped = hazards["ped_conflict"]
        occ = hazards["occupied_intersection"]
        pri = hazards["higher_priority"]
        simult = hazards["simultaneous_in_inter"]
        transition = hazards["transition_conflict"]

        interaction_pressure = max(ped, occ, pri, simult)
        free_flow = float(np.clip(1.0 - max(interaction_pressure, transition, colliding_close), 0.0, 1.0))
        caution = float(
            np.clip(
                max(interaction_pressure, transition)
                * np.clip((transition - self.CAUTION_TRANSITION_THRESHOLD) / 0.35, 0.0, 1.0),
                0.0,
                1.0,
            )
        )
        conflict_edge = float(
            np.clip(
                max(ego_at, 0.75 * ego_in)
                * max(interaction_pressure, transition)
                * np.clip((transition - self.EDGE_TRANSITION_THRESHOLD) / 0.30, 0.0, 1.0),
                0.0,
                1.0,
            )
        )
        critical = float(
            np.clip(
                max(
                    colliding_close,
                    simult,
                    ego_at * max(ped, pri, occ) * 0.9,
                    np.clip((transition - self.CRITICAL_TRANSITION_THRESHOLD) / 0.20, 0.0, 1.0),
                ),
                0.0,
                1.0,
            )
        )
        return {
            "free_flow": free_flow,
            "caution": caution,
            "conflict_edge": conflict_edge,
            "critical": critical,
        }

    def postprocess_action_scores(self, action_scores, fact_weights, obs_row=None):
        hazards = self._soft_rule_hazards(fact_weights)
        regimes = self._regime_scores(hazards, fact_weights)
        colliding_close = hazards["colliding_close"]
        ped = hazards["ped_conflict"]
        occ = hazards["occupied_intersection"]
        pri = hazards["higher_priority"]
        simult = hazards["simultaneous_in_inter"]
        transition_base = hazards["transition_conflict"]
        free_flow = regimes["free_flow"]
        caution = regimes["caution"]
        conflict_edge = regimes["conflict_edge"]
        critical = regimes["critical"]

        base_slow_risk = max(
            0.28 * colliding_close,
            0.12 * pri,
            0.16 * occ,
            0.24 * ped,
            0.52 * simult,
            0.34 * transition_base,
        )
        base_normal_risk = max(
            0.72 * colliding_close,
            0.42 * pri,
            0.52 * occ,
            0.64 * ped,
            0.86 * simult,
            0.82 * transition_base,
        )
        base_fast_risk = max(
            0.96 * colliding_close,
            0.88 * pri,
            0.94 * occ,
            0.98 * ped,
            1.00 * simult,
            1.00 * transition_base,
        )

        # Regime shaping:
        # free-flow: nearly flat
        # caution: fast loses safety first
        # conflict-edge: normal also starts losing safety
        # critical: stop strongly preferred, slow remains best moving action
        slow_risk = max(
            base_slow_risk,
            0.03 * free_flow,
            0.10 * caution,
            0.24 * conflict_edge,
            0.55 * critical,
        )
        normal_risk = max(
            base_normal_risk,
            0.04 * free_flow,
            0.48 * caution,
            0.82 * conflict_edge,
            0.97 * critical,
        )
        fast_risk = max(
            base_fast_risk,
            0.06 * free_flow,
            0.88 * caution,
            0.95 * conflict_edge,
            1.00 * critical,
        )

        soft_scores = [
            float(np.clip(1.0 - slow_risk, 0.0, 1.0)),
            float(np.clip(1.0 - normal_risk, 0.0, 1.0)),
            float(np.clip(1.0 - fast_risk, 0.0, 1.0)),
            1.0,
        ]
        return soft_scores
