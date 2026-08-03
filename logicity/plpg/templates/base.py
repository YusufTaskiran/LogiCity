import numpy as np


class BasePLPGTemplate:
    name = "base"

    def __init__(self, pred_grounding_index, rule_yaml_file=None, sensor_noise=None):
        self.pred_grounding_index = pred_grounding_index
        self.rule_yaml_file = rule_yaml_file
        self.sensor_noise = sensor_noise or {}
        self.stop_benign_safety = float(self.sensor_noise.get("stop_benign_safety", 0.8))

    def build_program_lines(self):
        raise NotImplementedError

    def obs_to_fact_weights(self, obs_row):
        raise NotImplementedError

    def fact_weights_to_metrics(self, fact_weights):
        return {}

    def _resolve_stop_benign_safety(self, obs_row=None):
        if obs_row is None:
            return float(getattr(self, "stop_benign_safety", 0.8))
        profile_values = self.sensor_noise.get("per_agent_stop_benign_safety")
        if not profile_values:
            return float(getattr(self, "stop_benign_safety", 0.8))
        profile_values = [float(value) for value in profile_values]
        obs_np = np.asarray(obs_row, dtype=np.float32).reshape(-1)
        if obs_np.size == 0:
            return float(getattr(self, "stop_benign_safety", 0.8))
        normed_agent_id = float(np.clip(obs_np[-1], 0.0, 1.0))
        if len(profile_values) == 1:
            return profile_values[0]
        agent_idx = int(round(normed_agent_id * float(len(profile_values) - 1)))
        agent_idx = min(max(agent_idx, 0), len(profile_values) - 1)
        return profile_values[agent_idx]

    def _resolve_agent_index(self, obs_row=None, profile_length=None):
        if obs_row is None:
            return 0
        obs_np = np.asarray(obs_row, dtype=np.float32).reshape(-1)
        if obs_np.size == 0:
            return 0
        if profile_length is None:
            profile_length = 1
        profile_length = max(int(profile_length), 1)
        normed_agent_id = float(np.clip(obs_np[-1], 0.0, 1.0))
        if profile_length == 1:
            return 0
        agent_idx = int(round(normed_agent_id * float(profile_length - 1)))
        return min(max(agent_idx, 0), profile_length - 1)

    def postprocess_action_scores(self, action_scores, fact_weights, obs_row=None):
        action_names = list(getattr(self, "action_names", []))
        if "stop" not in action_names or len(action_scores) <= 1:
            return action_scores
        if not bool(self.sensor_noise.get("enable_stop_benign_bias", False)):
            return action_scores

        stop_idx = action_names.index("stop")
        move_scores = [float(score) for idx, score in enumerate(action_scores) if idx != stop_idx]
        if len(move_scores) == 0:
            return action_scores

        # In the current templates, motion actions lose safety mass as hazard probability rises.
        # Use that soft hazard estimate to penalize unnecessary stopping in benign states while
        # preserving stop as the safest action when hazard is genuinely high.
        mean_move_safe = float(np.mean(move_scores))
        hazard_prob = float(np.clip(1.0 - mean_move_safe, 0.0, 1.0))
        stop_benign_safety = self._resolve_stop_benign_safety(obs_row=obs_row)
        stop_safe = stop_benign_safety + (1.0 - stop_benign_safety) * hazard_prob

        adjusted_scores = list(action_scores)
        adjusted_scores[stop_idx] = min(float(adjusted_scores[stop_idx]), float(stop_safe))
        return adjusted_scores

    def _apply_sensor_noise(self, fact_value):
        mode = self.sensor_noise.get("mode")
        epsilon = float(self.sensor_noise.get("epsilon", 0.0))
        epsilon = min(max(epsilon, 0.0), 0.5)
        if mode == "soft_symmetric":
            return (1.0 - epsilon) if fact_value > 0.5 else epsilon
        if mode == "bit_flip":
            binary_value = 1.0 if fact_value > 0.5 else 0.0
            if np.random.rand() < epsilon:
                return 1.0 - binary_value
            return binary_value
        return fact_value

    def _binary_fact(self, value):
        return self._apply_sensor_noise(float(value > 0.5))

    def _prob_fact(self, value):
        return self._apply_sensor_noise(float(np.clip(value, 0.0, 1.0)))
