import argparse
import copy
import pickle as pkl
from pathlib import Path

import yaml

from logicity.utils.load import CityLoader
from logicity.utils.pred_converter import z3 as predz3
from logicity.utils.vis import visualize_city


def parse_xy(value):
    x_str, y_str = value.split(",")
    return [int(x_str), int(y_str)]


def build_parser():
    parser = argparse.ArgumentParser(description="Create a custom easy multi-agent episode.")
    parser.add_argument(
        "--base-episodes",
        default="log_rl/easy_lite_test_episodes.pkl",
        help="Existing episode cache used as a template.",
    )
    parser.add_argument(
        "--config",
        default="config/tasks/Nav/easy/algo/multi_shared_plpg_test_k2.yaml",
        help="Task config used to validate the generated episode.",
    )
    parser.add_argument(
        "--output",
        default="log_rl/easy_symmetric_equal_priority_episode.pkl",
        help="Where to write the generated single-episode cache.",
    )
    parser.add_argument("--car1-start", type=parse_xy, default=[40, 46])
    parser.add_argument("--car1-goal", type=parse_xy, default=[56, 18])
    parser.add_argument("--car2-start", type=parse_xy, default=[46, 35])
    parser.add_argument("--car2-goal", type=parse_xy, default=[46, 62])
    parser.add_argument("--equal-priority", type=int, default=2)
    parser.add_argument("--bg-car-start", type=parse_xy, default=[10, 34])
    parser.add_argument("--bg-car-goal", type=parse_xy, default=[37, 10])
    parser.add_argument(
        "--static-image",
        default="rollout_vis/easy_symmetric_equal_priority_episode_init.png",
        help="Path for a static reset-state render.",
    )
    return parser


def set_agent(agent_info, start, goal, priority=None):
    agent_info["start"] = list(start)
    agent_info["goal"] = list(goal)
    agent_info["pos"] = list(start)
    if priority is not None:
        agent_info["priority"] = int(priority)
        agent_info["concepts"]["priority"] = int(priority)


def local_predicate_report(city, rl_agent_ids):
    layerid2listid = {agent.layer_id: idx for idx, agent in enumerate(city.agents)}
    _, partial_agents, partial_world, partial_intersections, _ = city.local_planner.break_world_matrix(
        city.city_grid.clone(),
        city.agents,
        city.intersection_matrix.clone(),
        layerid2listid,
        rl_agent=rl_agent_ids,
    )
    for ego_name in sorted(partial_agents.keys()):
        if ego_name not in ("Car_3", "Car_5"):
            continue
        pa = partial_agents[ego_name]
        pw = partial_world[ego_name]
        pi = partial_intersections[ego_name]
        ego_key = next(key for key in pa if key.startswith("ego_"))
        ego = pa[ego_key]
        ego_entity = f"Entity_{ego.type}_{ego.layer_id}"
        other_entities = []
        for key, value in pa.items():
            if key.startswith("ego_") or key.startswith("PH_"):
                continue
            other_entities.append(f"Entity_{value.type}_{value.layer_id}")
        print(f"[{ego_name}]")
        print(f"  ego_entity={ego_entity}")
        print(f"  other_entities={other_entities}")
        print(f"  IsAtInter={predz3.IsAtInter(pw, pi, pa, ego_entity)}")
        print(f"  IsInInter={predz3.IsInInter(pw, pi, pa, ego_entity)}")
        for other in other_entities:
            print(
                "  against {}: HigherPri={} CollidingClose={}".format(
                    other,
                    predz3.HigherPri(pw, pi, pa, ego_entity, other),
                    predz3.CollidingClose(pw, pi, pa, ego_entity, other),
                )
            )


def main():
    args = build_parser().parse_args()

    with open(args.base_episodes, "rb") as f:
        episodes = pkl.load(f)
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    episode = copy.deepcopy(episodes[0])
    set_agent(
        episode["agents"]["Car_1"],
        args.car1_start,
        args.car1_goal,
        priority=args.equal_priority,
    )
    set_agent(
        episode["agents"]["Car_2"],
        args.car2_start,
        args.car2_goal,
        priority=args.equal_priority,
    )
    set_agent(
        episode["agents"]["Car_3"],
        args.bg_car_start,
        args.bg_car_goal,
    )
    episode["label_info"] = {
        "oracle_step": 40,
        "scenario": "equal_priority_intersection_probe",
        "notes": {
            "car1": {"start": args.car1_start, "goal": args.car1_goal},
            "car2": {"start": args.car2_start, "goal": args.car2_goal},
            "priority": args.equal_priority,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pkl.dump({0: episode}, f)

    print(f"Wrote episode to {output_path}")
    city, _ = CityLoader.from_yaml(**config["simulation"], episode_cache=episode)
    static_image_path = Path(args.static_image)
    static_image_path.parent.mkdir(parents=True, exist_ok=True)
    visualize_city(city, 4 * city.grid_size[0], -1, str(static_image_path))
    print(f"Wrote static image to {static_image_path}")
    print("Validation report:")
    local_predicate_report(city, rl_agent_ids={3, 5})


if __name__ == "__main__":
    main()
