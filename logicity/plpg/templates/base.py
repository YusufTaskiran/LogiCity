import numpy as np


class BasePLPGTemplate:
    name = "base"

    def __init__(self, pred_grounding_index, rule_yaml_file=None, sensor_noise=None):
        self.pred_grounding_index = pred_grounding_index
        self.rule_yaml_file = rule_yaml_file
        self.sensor_noise = sensor_noise or {}

    def build_program_lines(self):
        raise NotImplementedError

    def obs_to_fact_weights(self, obs_row):
        raise NotImplementedError

    def fact_weights_to_metrics(self, fact_weights):
        return {}

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
