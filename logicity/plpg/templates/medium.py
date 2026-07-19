import torch as th

from logicity.plpg.translator import translate_rule_yaml_to_hazard_rules

from .base import BasePLPGTemplate


class MediumPLPGTemplate(BasePLPGTemplate):
    name = "medium"
    action_names = ["slow", "normal", "fast", "stop"]

    def __init__(self, pred_grounding_index, rule_yaml_file=None, sensor_noise=None):
        super().__init__(pred_grounding_index, rule_yaml_file=rule_yaml_file, sensor_noise=sensor_noise)
        self.num_entities = pred_grounding_index["IsAtInter"][1] - pred_grounding_index["IsAtInter"][0]
        self._is_at_inter = pred_grounding_index["IsAtInter"]
        self._is_in_inter = pred_grounding_index["IsInInter"]
        self._same_inter = pred_grounding_index["SameInter"]
        self._higher_pri = pred_grounding_index["HigherPri"]
        self._colliding_close = pred_grounding_index["CollidingClose"]
        self._is_ambulance = pred_grounding_index["IsAmbulance"]
        self._is_old = pred_grounding_index["IsOld"]
        self._is_bus = pred_grounding_index["IsBus"]
        self._is_pedestrian = pred_grounding_index["IsPedestrian"]
        self._right_of = pred_grounding_index["RightOf"]
        self._next_to = pred_grounding_index["NextTo"]
        self._obs_logic_end = max(
            self._is_at_inter[1],
            self._is_in_inter[1],
            self._same_inter[1],
            self._higher_pri[1],
            self._colliding_close[1],
            self._is_ambulance[1],
            self._is_old[1],
            self._is_bus[1],
            self._is_pedestrian[1],
            self._right_of[1],
            self._next_to[1],
        )

    def _pair_index(self, i, j):
        return i * self.num_entities + j

    def build_program_lines(self):
        lines = [
            "0.25::act(slow); 0.25::act(normal); 0.25::act(fast); 0.25::act(stop).",
            "0.0::ego_is_ambulance.",
            "0.0::ego_is_old.",
            "0.0::ego_is_bus.",
            "0.0::ego_is_at_inter.",
            "0.0::ego_is_in_inter.",
        ]
        for i in range(1, self.num_entities):
            lines.append(f"0.0::other_{i}_is_in_inter.")
            lines.append(f"0.0::other_{i}_same_inter.")
            lines.append(f"0.0::other_{i}_is_at_inter.")
            lines.append(f"0.0::other_{i}_higher_pri.")
            lines.append(f"0.0::other_{i}_colliding_close.")
            lines.append(f"0.0::other_{i}_is_ambulance.")
            lines.append(f"0.0::other_{i}_is_old.")
            lines.append(f"0.0::other_{i}_is_pedestrian.")
            lines.append(f"0.0::other_{i}_right_of_ego.")
            lines.append(f"0.0::other_{i}_next_to_ego.")

        if self.rule_yaml_file:
            lines.extend(translate_rule_yaml_to_hazard_rules(self.rule_yaml_file, self.num_entities, self.name))
        else:
            for i in range(1, self.num_entities):
                lines.append(
                    f"hazard :- \\+ ego_is_ambulance, \\+ ego_is_old, ego_is_at_inter, other_{i}_same_inter, other_{i}_is_in_inter."
                )
                lines.append(
                    f"hazard :- \\+ ego_is_ambulance, \\+ ego_is_old, ego_is_at_inter, other_{i}_same_inter, other_{i}_is_at_inter, other_{i}_higher_pri."
                )
                lines.append(
                    f"hazard :- \\+ ego_is_ambulance, \\+ ego_is_old, ego_is_in_inter, other_{i}_same_inter, other_{i}_is_in_inter, other_{i}_is_ambulance."
                )
                lines.append(
                    f"hazard :- ego_is_bus, \\+ ego_is_in_inter, \\+ ego_is_at_inter, other_{i}_right_of_ego, other_{i}_next_to_ego, other_{i}_is_pedestrian."
                )
                lines.append(
                    f"hazard :- ego_is_ambulance, other_{i}_right_of_ego, other_{i}_is_old."
                )
                lines.append(
                    f"hazard :- \\+ ego_is_ambulance, \\+ ego_is_old, other_{i}_colliding_close."
                )

        lines.extend(
            [
                "unsafe :- hazard, act(slow).",
                "unsafe :- hazard, act(normal).",
                "unsafe :- hazard, act(fast).",
                "safe :- \\+ unsafe.",
                "query(safe).",
                "query(act(slow)).",
                "query(act(normal)).",
                "query(act(fast)).",
                "query(act(stop)).",
            ]
        )
        return lines

    def obs_to_fact_weights(self, obs_row: th.Tensor):
        obs_logic = obs_row[: self._obs_logic_end]
        fact_weights = {
            "ego_is_ambulance": self._prob_fact(obs_logic[self._is_ambulance[0]].item()),
            "ego_is_old": self._prob_fact(obs_logic[self._is_old[0]].item()),
            "ego_is_bus": self._prob_fact(obs_logic[self._is_bus[0]].item()),
            "ego_is_at_inter": self._prob_fact(obs_logic[self._is_at_inter[0]].item()),
            "ego_is_in_inter": self._prob_fact(obs_logic[self._is_in_inter[0]].item()),
        }
        for i in range(1, self.num_entities):
            fact_weights[f"other_{i}_is_in_inter"] = self._prob_fact(obs_logic[self._is_in_inter[0] + i].item())
            fact_weights[f"other_{i}_same_inter"] = self._prob_fact(
                obs_logic[self._same_inter[0] + self._pair_index(0, i)].item()
            )
            fact_weights[f"other_{i}_is_at_inter"] = self._prob_fact(obs_logic[self._is_at_inter[0] + i].item())
            fact_weights[f"other_{i}_higher_pri"] = self._prob_fact(
                obs_logic[self._higher_pri[0] + self._pair_index(i, 0)].item()
            )
            fact_weights[f"other_{i}_colliding_close"] = self._prob_fact(
                obs_logic[self._colliding_close[0] + self._pair_index(0, i)].item()
            )
            fact_weights[f"other_{i}_is_ambulance"] = self._prob_fact(obs_logic[self._is_ambulance[0] + i].item())
            fact_weights[f"other_{i}_is_old"] = self._prob_fact(obs_logic[self._is_old[0] + i].item())
            fact_weights[f"other_{i}_is_pedestrian"] = self._prob_fact(
                obs_logic[self._is_pedestrian[0] + i].item()
            )
            fact_weights[f"other_{i}_right_of_ego"] = self._prob_fact(
                obs_logic[self._right_of[0] + self._pair_index(i, 0)].item()
            )
            fact_weights[f"other_{i}_next_to_ego"] = self._prob_fact(
                obs_logic[self._next_to[0] + self._pair_index(i, 0)].item()
            )
        return fact_weights

    def fact_weights_to_metrics(self, fact_weights):
        hazard = 0.0
        for i in range(1, self.num_entities):
            if (
                (
                    fact_weights["ego_is_ambulance"] <= 0.5
                    and fact_weights["ego_is_old"] <= 0.5
                    and fact_weights["ego_is_at_inter"] > 0.5
                    and fact_weights[f"other_{i}_same_inter"] > 0.5
                    and fact_weights[f"other_{i}_is_in_inter"] > 0.5
                )
                or (
                    fact_weights["ego_is_ambulance"] <= 0.5
                    and fact_weights["ego_is_old"] <= 0.5
                    and fact_weights["ego_is_at_inter"] > 0.5
                    and fact_weights[f"other_{i}_same_inter"] > 0.5
                    and fact_weights[f"other_{i}_is_at_inter"] > 0.5
                    and fact_weights[f"other_{i}_higher_pri"] > 0.5
                )
                or (
                    fact_weights["ego_is_ambulance"] <= 0.5
                    and fact_weights["ego_is_old"] <= 0.5
                    and fact_weights["ego_is_in_inter"] > 0.5
                    and fact_weights[f"other_{i}_same_inter"] > 0.5
                    and fact_weights[f"other_{i}_is_in_inter"] > 0.5
                    and fact_weights[f"other_{i}_is_ambulance"] > 0.5
                )
                or (
                    fact_weights["ego_is_bus"] > 0.5
                    and fact_weights["ego_is_in_inter"] <= 0.5
                    and fact_weights["ego_is_at_inter"] <= 0.5
                    and fact_weights[f"other_{i}_right_of_ego"] > 0.5
                    and fact_weights[f"other_{i}_next_to_ego"] > 0.5
                    and fact_weights[f"other_{i}_is_pedestrian"] > 0.5
                )
                or (
                    fact_weights["ego_is_ambulance"] > 0.5
                    and fact_weights[f"other_{i}_right_of_ego"] > 0.5
                    and fact_weights[f"other_{i}_is_old"] > 0.5
                )
                or (
                    fact_weights["ego_is_ambulance"] <= 0.5
                    and fact_weights["ego_is_old"] <= 0.5
                    and fact_weights[f"other_{i}_colliding_close"] > 0.5
                )
            ):
                hazard = 1.0
                break
        return {"hazard": hazard}
