from .easy import EasyPLPGTemplate
from .hard import HardPLPGTemplate
from .medium import MediumPLPGTemplate


TEMPLATE_REGISTRY = {
    "easy": EasyPLPGTemplate,
    "medium": MediumPLPGTemplate,
    "hard": HardPLPGTemplate,
}


def build_template(template_name, pred_grounding_index, rule_yaml_file=None, sensor_noise=None):
    if template_name not in TEMPLATE_REGISTRY:
        raise ValueError("Unknown PLPG template '{}'.".format(template_name))
    template_cls = TEMPLATE_REGISTRY[template_name]
    return template_cls(pred_grounding_index, rule_yaml_file=rule_yaml_file, sensor_noise=sensor_noise)
