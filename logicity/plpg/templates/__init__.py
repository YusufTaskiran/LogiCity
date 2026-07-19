from .easy import EasyPLPGTemplate
from .easy_appropriate import EasyAppropriatePLPGTemplate
from .easy_fine import EasyFinePLPGTemplate
from .hard import HardPLPGTemplate
from .hard_fine import HardFinePLPGTemplate
from .medium import MediumPLPGTemplate
from .medium_fine import MediumFinePLPGTemplate


TEMPLATE_REGISTRY = {
    "easy": EasyPLPGTemplate,
    "easy_appropriate": EasyAppropriatePLPGTemplate,
    "easy_fine": EasyFinePLPGTemplate,
    "medium": MediumPLPGTemplate,
    "medium_fine": MediumFinePLPGTemplate,
    "hard": HardPLPGTemplate,
    "hard_fine": HardFinePLPGTemplate,
}


def build_template(template_name, pred_grounding_index, rule_yaml_file=None, sensor_noise=None):
    if template_name not in TEMPLATE_REGISTRY:
        raise ValueError("Unknown PLPG template '{}'.".format(template_name))
    template_cls = TEMPLATE_REGISTRY[template_name]
    return template_cls(pred_grounding_index, rule_yaml_file=rule_yaml_file, sensor_noise=sensor_noise)
