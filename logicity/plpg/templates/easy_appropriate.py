import torch as th

from logicity.plpg.translator import translate_rule_yaml_to_hazard_rules

from .base import BasePLPGTemplate


class EasyAppropriatePLPGTemplate(BasePLPGTemplate):
    name = "easy_appropriate"
    action_names = ["slow", "normal", "fast", "stop"]

    def __init__(self, pred_grounding_index, rule_yaml_file=None, sensor_noise=None):
        super().__init__(pred_grounding_index, rule_yaml_file=rule_yaml_file, sensor_noise=sensor_noise)
        self.num_entities = pred_grounding_index["IsAtInter"][1] - pred_grounding_index["IsAtInter"][0]
        self._is_at_inter = pred_grounding_index["IsAtInter"]
        self._is_in_inter = pred_grounding_index["IsInInter"]
        self._higher_pri = pred_grounding_index["HigherPri"]
        self._colliding_close = pred_grounding_index["CollidingClose"]

    def _pair_index(self, i, j):
        return i * self.num_entities + j

    def build_program_lines(self):
        lines = [
            "0.25::act(slow); 0.25::act(normal); 0.25::act(fast); 0.25::act(stop).",
            "0.0::is_at_inter_ego.",
        ]
        for i in range(1, self.num_entities):
            lines.append(f"0.0::is_in_inter_{i}.")
            lines.append(f"0.0::higher_pri_{i}.")
            lines.append(f"0.0::colliding_close_{i}.")

        if self.rule_yaml_file:
            lines.extend(translate_rule_yaml_to_hazard_rules(self.rule_yaml_file, self.num_entities, "easy"))
        else:
            for i in range(1, self.num_entities):
                lines.append(f"hazard :- is_at_inter_ego, is_in_inter_{i}.")
                lines.append(f"hazard :- is_at_inter_ego, higher_pri_{i}.")
                lines.append(f"hazard :- colliding_close_{i}.")

        lines.extend(
            [
                "need_to_stop :- hazard.",
                "okay_to_move :- \\+ need_to_stop.",
                "appropriate :- act(stop), need_to_stop.",
                "appropriate :- act(slow), okay_to_move.",
                "appropriate :- act(normal), okay_to_move.",
                "appropriate :- act(fast), okay_to_move.",
                "safe :- appropriate.",
                "query(safe).",
                "query(act(slow)).",
                "query(act(normal)).",
                "query(act(fast)).",
                "query(act(stop)).",
            ]
        )
        return lines

    def obs_to_fact_weights(self, obs_row: th.Tensor):
        obs_logic = obs_row[: self._colliding_close[1]]
        fact_weights = {"is_at_inter_ego": self._prob_fact(obs_logic[self._is_at_inter[0]].item())}

        for i in range(1, self.num_entities):
            fact_weights[f"is_in_inter_{i}"] = self._prob_fact(obs_logic[self._is_in_inter[0] + i].item())
            fact_weights[f"higher_pri_{i}"] = self._prob_fact(
                obs_logic[self._higher_pri[0] + self._pair_index(i, 0)].item()
            )
            fact_weights[f"colliding_close_{i}"] = self._prob_fact(
                obs_logic[self._colliding_close[0] + self._pair_index(0, i)].item()
            )
        return fact_weights

    def fact_weights_to_metrics(self, fact_weights):
        hazard = 0.0
        for i in range(1, self.num_entities):
            if (
                (fact_weights["is_at_inter_ego"] > 0.5 and fact_weights[f"is_in_inter_{i}"] > 0.5)
                or (fact_weights["is_at_inter_ego"] > 0.5 and fact_weights[f"higher_pri_{i}"] > 0.5)
                or (fact_weights[f"colliding_close_{i}"] > 0.5)
            ):
                hazard = 1.0
                break
        return {"hazard": hazard}
