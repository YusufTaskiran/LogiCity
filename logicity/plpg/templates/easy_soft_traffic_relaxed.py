import numpy as np

from .easy_soft_traffic import EasySoftTrafficPLPGTemplate


class EasySoftTrafficRelaxedPLPGTemplate(EasySoftTrafficPLPGTemplate):
    name = "easy_soft_traffic_relaxed"

    # Lower the contribution of each soft-traffic cue to overall hazard.
    RULE_WEIGHT_MAP = {
        "colliding_close": 0.75,
        "ped_conflict": 0.62,
        "occupied_intersection": 0.52,
        "higher_priority": 0.52,
        "simultaneous_in_inter": 0.68,
        "transition_conflict": 0.45,
    }

    # Compress the motion-action safety gap more aggressively so fast/normal/slow stay close.
    MOVE_SCORE_BLEND = 0.80

    # Keep stop useful in clearly critical states, but let motion dominate much more often.
    STOP_MARGIN_BENIGN = -0.02
    STOP_MARGIN_CRITICAL = 0.08

    def _soft_rule_hazards(self, fact_weights):
        base_hazards = super()._soft_rule_hazards(fact_weights)
        scaled = {}
        for key, value in base_hazards.items():
            if key == "hazard":
                continue
            weight = float(self.RULE_WEIGHT_MAP.get(key, 1.0))
            scaled[key] = float(np.clip(weight * float(value), 0.0, 1.0))
        scaled["hazard"] = float(np.clip(max(scaled.values(), default=0.0), 0.0, 1.0))
        return scaled

    def postprocess_action_scores(self, action_scores, fact_weights, obs_row=None):
        relaxed_scores = list(super().postprocess_action_scores(action_scores, fact_weights, obs_row=obs_row))
        hazards = self._soft_rule_hazards(fact_weights)
        hazard = float(hazards["hazard"])

        move_scores = np.asarray(relaxed_scores[:3], dtype=np.float64)
        move_mean = float(move_scores.mean())
        blended_moves = move_scores * (1.0 - self.MOVE_SCORE_BLEND) + move_mean * self.MOVE_SCORE_BLEND

        best_move = float(np.max(blended_moves))
        stop_margin = self.STOP_MARGIN_BENIGN + (self.STOP_MARGIN_CRITICAL - self.STOP_MARGIN_BENIGN) * hazard
        stop_score = min(float(relaxed_scores[3]), float(np.clip(best_move + stop_margin, 0.0, 1.0)))

        return [
            float(np.clip(blended_moves[0], 0.0, 1.0)),
            float(np.clip(blended_moves[1], 0.0, 1.0)),
            float(np.clip(blended_moves[2], 0.0, 1.0)),
            stop_score,
        ]
