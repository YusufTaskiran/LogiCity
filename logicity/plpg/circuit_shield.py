import torch as th

from problog import get_evaluatable
from problog.program import PrologString

from .templates import build_template


class ProbLogCircuitShield:
    def __init__(self, template_name, pred_grounding_index, rule_yaml_file=None, sensor_noise=None):
        self.template = build_template(
            template_name,
            pred_grounding_index,
            rule_yaml_file=rule_yaml_file,
            sensor_noise=sensor_noise,
        )
        self.action_names = list(self.template.action_names)
        self._compiled = self._compile_program()

    def _compile_program(self):
        program = PrologString("\n".join(self.template.build_program_lines()))
        compiled = get_evaluatable(name="ddnnf").create_from(program)
        atom_ids = {}
        for name, key, _label in compiled.get_names_with_label():
            atom_ids[str(name)] = key
        return {
            "circuit": compiled,
            "base_weights": dict(compiled.get_weights()),
            "atom_ids": atom_ids,
        }

    def _evaluate_safe_query(self, action_probs, fact_weights):
        weights = dict(self._compiled["base_weights"])
        atom_ids = self._compiled["atom_ids"]
        for action_name, action_prob in zip(self.action_names, action_probs):
            weights[atom_ids[f"act({action_name})"]] = float(action_prob)
        for fact_name, fact_value in fact_weights.items():
            atom_id = atom_ids.get(fact_name)
            if atom_id is not None:
                weights[atom_id] = float(fact_value)
        result = self._compiled["circuit"].evaluate(weights=weights)
        for query_name, query_value in result.items():
            if str(query_name) == "safe":
                return float(query_value)
        raise KeyError("Query 'safe' not found in ProbLog evaluation result: {}".format(result))

    def compute_safety_probs(self, obs_tensor: th.Tensor) -> th.Tensor:
        safety_probs = []
        metric_acc = {}
        one_hot_actions = []
        for i in range(len(self.action_names)):
            row = [0.0] * len(self.action_names)
            row[i] = 1.0
            one_hot_actions.append(row)
        for obs_row in obs_tensor.detach().cpu():
            fact_weights = self.template.obs_to_fact_weights(obs_row)
            for key, value in self.template.fact_weights_to_metrics(fact_weights).items():
                metric_acc.setdefault(key, []).append(value)
            action_scores = [self._evaluate_safe_query(action_prob, fact_weights) for action_prob in one_hot_actions]
            action_scores = self.template.postprocess_action_scores(action_scores, fact_weights, obs_row=obs_row)
            safety_probs.append(action_scores)
        return (
            th.tensor(safety_probs, dtype=obs_tensor.dtype, device=obs_tensor.device),
            {
                key: th.tensor(values, dtype=obs_tensor.dtype, device=obs_tensor.device)
                for key, values in metric_acc.items()
            },
        )

    def shield_policy(self, base_probs: th.Tensor, obs_tensor: th.Tensor):
        safety_probs, template_metrics = self.compute_safety_probs(obs_tensor)
        weighted = base_probs * safety_probs
        denom = weighted.sum(dim=1, keepdim=True).clamp_min(1e-8)
        shielded_probs = weighted / denom
        base_policy_safe_prob = weighted.sum(dim=1).clamp_min(1e-8)
        shielded_policy_safe_prob = (shielded_probs * safety_probs).sum(dim=1).clamp_min(1e-8)
        return {
            "safety_probs": safety_probs,
            "shielded_probs": shielded_probs,
            "base_policy_safe_prob": base_policy_safe_prob,
            "shielded_policy_safe_prob": shielded_policy_safe_prob,
            "template_metrics": template_metrics,
        }
