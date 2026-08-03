import yaml


def _load_task_formula(rule_yaml_file):
    with open(rule_yaml_file, "r") as handle:
        data = yaml.safe_load(handle) or {}
    task_rules = data.get("Rules", {}).get("Task", [])
    if not task_rules:
        raise ValueError("No Task rules found in {}".format(rule_yaml_file))
    return "\n".join(rule.get("formula", "") for rule in task_rules)


def _require_tokens(formula, rule_yaml_file, tokens, mode_name):
    missing = [token for token in tokens if token not in formula]
    if missing:
        raise NotImplementedError(
            "Automatic translation for '{}' failed on {}. Missing expected tokens: {}".format(
                mode_name, rule_yaml_file, ", ".join(missing)
            )
        )


def translate_easy_rule_yaml_to_hazard_rules(rule_yaml_file, num_entities):
    formula = _load_task_formula(rule_yaml_file)
    has_collision_only_tokens = all(
        token in formula for token in ["CollidingClose", "Stop(entity)"]
    )
    has_full_easy_tokens = all(
        token in formula
        for token in ["IsAtInter", "IsInInter", "SameInter", "HigherPri", "CollidingClose", "IsPedestrian", "Stop(entity)"]
    )
    lines = []
    if has_collision_only_tokens and not has_full_easy_tokens:
        for i in range(1, num_entities):
            lines.append(f"hazard :- other_{i}_colliding_close.")
        return lines
    _require_tokens(
        formula,
        rule_yaml_file,
        ["IsAtInter", "IsInInter", "SameInter", "HigherPri", "CollidingClose", "IsPedestrian", "Stop(entity)"],
        "easy",
    )
    for i in range(1, num_entities):
        lines.append(f"hazard :- ego_is_at_inter, other_{i}_same_inter, other_{i}_is_in_inter.")
        lines.append(f"hazard :- ego_is_at_inter, other_{i}_same_inter, other_{i}_is_at_inter, other_{i}_higher_pri.")
        lines.append(f"hazard :- ego_is_car, other_{i}_is_pedestrian, other_{i}_same_inter, ego_is_at_inter, other_{i}_is_at_inter.")
        lines.append(f"hazard :- ego_is_car, other_{i}_is_pedestrian, other_{i}_same_inter, ego_is_at_inter, other_{i}_is_in_inter.")
        lines.append(f"hazard :- ego_is_car, other_{i}_is_pedestrian, other_{i}_same_inter, ego_is_in_inter, other_{i}_is_at_inter.")
        lines.append(f"hazard :- ego_is_car, other_{i}_is_pedestrian, other_{i}_same_inter, ego_is_in_inter, other_{i}_is_in_inter.")
        lines.append(f"hazard :- other_{i}_colliding_close.")
    return lines


def translate_medium_rule_yaml_to_hazard_rules(rule_yaml_file, num_entities):
    formula = _load_task_formula(rule_yaml_file)
    _require_tokens(
        formula,
        rule_yaml_file,
        [
            "Not(IsAmbulance(entity))",
            "Not(IsOld(entity))",
            "IsAtInter(entity)",
            "IsInInter(dummyEntityA)",
            "SameInter(entity, dummyEntityA)",
            "HigherPri(dummyEntityA, entity)",
            "IsAmbulance(dummyEntityA)",
            "IsBus(entity)",
            "RightOf(dummyEntityA, entity)",
            "NextTo(dummyEntityA, entity)",
            "IsPedestrian(dummyEntityA)",
            "IsOld(dummyEntityA)",
            "CollidingClose(entity, dummyEntityA)",
            "Stop(entity)",
        ],
        "medium",
    )
    lines = []
    for i in range(1, num_entities):
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
    return lines


def translate_hard_rule_yaml_to_hazard_rules(rule_yaml_file, num_entities):
    formula = _load_task_formula(rule_yaml_file)
    _require_tokens(
        formula,
        rule_yaml_file,
        [
            "Not(IsAmbulance(entity))",
            "Not(IsOld(entity))",
            "IsAtInter(entity)",
            "IsInInter(dummyEntityA)",
            "SameInter(entity, dummyEntityA)",
            "HigherPri(dummyEntityA, entity)",
            "IsAmbulance(dummyEntityA)",
            "Not(IsPolice(entity))",
            "IsCar(entity)",
            "LeftOf(dummyEntityA, entity)",
            "IsClose(dummyEntityA, entity)",
            "IsPolice(dummyEntityA)",
            "IsBus(entity)",
            "RightOf(dummyEntityA, entity)",
            "NextTo(dummyEntityA, entity)",
            "IsPedestrian(dummyEntityA)",
            "IsOld(dummyEntityA)",
            "CollidingClose(entity, dummyEntityA)",
            "Stop(entity)",
        ],
        "hard",
    )
    lines = []
    for i in range(1, num_entities):
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
            f"hazard :- \\+ ego_is_ambulance, \\+ ego_is_police, ego_is_car, \\+ ego_is_in_inter, \\+ ego_is_at_inter, other_{i}_left_of_ego, other_{i}_is_close_to_ego, other_{i}_is_police."
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
    return lines


def translate_rule_yaml_to_hazard_rules(rule_yaml_file, num_entities, mode):
    mode = str(mode).lower()
    if mode == "easy":
        return translate_easy_rule_yaml_to_hazard_rules(rule_yaml_file, num_entities)
    if mode == "medium":
        return translate_medium_rule_yaml_to_hazard_rules(rule_yaml_file, num_entities)
    if mode == "hard":
        return translate_hard_rule_yaml_to_hazard_rules(rule_yaml_file, num_entities)
    raise ValueError("Unknown rule translation mode '{}'.".format(mode))
