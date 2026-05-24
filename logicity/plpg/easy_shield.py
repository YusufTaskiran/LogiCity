from .circuit_shield import ProbLogCircuitShield


class EasySafePathPLPGShield(ProbLogCircuitShield):
    def __init__(self, pred_grounding_index, rule_yaml_file=None):
        super().__init__("easy", pred_grounding_index, rule_yaml_file=rule_yaml_file)
