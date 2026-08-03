from .easy_soft_traffic import EasySoftTrafficPLPGTemplate
from .easy_soft_traffic_relaxed import EasySoftTrafficRelaxedPLPGTemplate


class EasySoftTrafficHybridPLPGTemplate(EasySoftTrafficPLPGTemplate):
    name = "easy_soft_traffic_hybrid"

    def __init__(self, pred_grounding_index, rule_yaml_file=None, sensor_noise=None):
        super().__init__(pred_grounding_index, rule_yaml_file=rule_yaml_file, sensor_noise=sensor_noise)
        shared_sensor_noise = dict(sensor_noise or {})
        self._conservative_template = EasySoftTrafficPLPGTemplate(
            pred_grounding_index,
            rule_yaml_file=rule_yaml_file,
            sensor_noise=shared_sensor_noise,
        )
        self._aggressive_template = EasySoftTrafficRelaxedPLPGTemplate(
            pred_grounding_index,
            rule_yaml_file=rule_yaml_file,
            sensor_noise=shared_sensor_noise,
        )

    def _resolve_profile_mode(self, obs_row=None):
        profile = list(self.sensor_noise.get("per_agent_shield_profile") or [])
        if len(profile) == 0:
            return "conservative"
        agent_idx = self._resolve_agent_index(obs_row=obs_row, profile_length=len(profile))
        mode = str(profile[agent_idx]).strip().lower()
        if mode in {"aggressive", "aggr", "relaxed", "agr"}:
            return "aggressive"
        return "conservative"

    def postprocess_action_scores(self, action_scores, fact_weights, obs_row=None):
        mode = self._resolve_profile_mode(obs_row=obs_row)
        if mode == "aggressive":
            return self._aggressive_template.postprocess_action_scores(
                action_scores,
                fact_weights,
                obs_row=obs_row,
            )
        return self._conservative_template.postprocess_action_scores(
            action_scores,
            fact_weights,
            obs_row=obs_row,
        )
