import argparse
import pickle as pkl
from pathlib import Path

import torch
import yaml

from logicity.core.config import STREET_ID, TYPE_MAP
from logicity.utils.load import CityLoader
from logicity.utils.pred_converter import z3 as predz3
from logicity.utils.vis import visualize_city


def build_parser():
    parser = argparse.ArgumentParser(description="Create a two-car symmetric intersection episode.")
    parser.add_argument(
        "--config",
        default="config/tasks/Nav/easy/algo/multi_shared_plpg_test_two_car_symmetric.yaml",
        help="Config used both for search and later evaluation.",
    )
    parser.add_argument(
        "--output",
        default="log_rl/easy_two_car_symmetric_episode.pkl",
        help="Single-episode cache output path.",
    )
    parser.add_argument(
        "--static-image",
        default="rollout_vis/easy_two_car_symmetric_episode_init.png",
        help="Static render of the initial state.",
    )
    parser.add_argument(
        "--max_offset",
        type=int,
        default=16,
        help="Maximum distance from the center intersection boundary to search.",
    )
    return parser


def get_center_intersection_mask(city):
    intersection_blocks = city.intersection_matrix[2]
    labels = torch.unique(intersection_blocks)
    labels = labels[labels > 0]
    if labels.numel() == 0:
        raise RuntimeError("No intersection labels found on the map.")
    grid_center = torch.tensor(
        [intersection_blocks.shape[0] / 2.0, intersection_blocks.shape[1] / 2.0],
        dtype=torch.float32,
    )
    best_label = None
    best_distance = None
    for label in labels.tolist():
        coords = (intersection_blocks == label).nonzero(as_tuple=False).float()
        centroid = coords.mean(dim=0)
        distance = torch.norm(centroid - grid_center).item()
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_label = label
    return intersection_blocks == best_label


def first_intersection_index(agent, mask):
    coords = agent.global_traj.long()
    hits = mask[coords[:, 0], coords[:, 1]].nonzero(as_tuple=False)
    if hits.numel() == 0:
        return None
    return int(hits[0].item())


def last_intersection_index(agent, mask):
    coords = agent.global_traj.long()
    hits = mask[coords[:, 0], coords[:, 1]].nonzero(as_tuple=False)
    if hits.numel() == 0:
        return None
    return int(hits[-1].item())


def step_direction(traj, idx_from, idx_to):
    delta = traj[idx_to] - traj[idx_from]
    dx = int(delta[0].item())
    dy = int(delta[1].item())
    if abs(dx) >= abs(dy):
        return "down" if dx > 0 else "up"
    return "right" if dy > 0 else "left"


def build_episode(car1_start, car1_goal, car2_start, car2_goal):
    return {
        "city_grid": None,
        "agents": {
            "Car_1": {
                "start": list(car1_start),
                "goal": list(car1_goal),
                "concepts": {"type": "Car", "priority": 2},
                "type": "Car",
                "priority": 2,
                "pos": list(car1_start),
                "layer_id": 3,
                "id": 1,
            },
            "Car_2": {
                "start": list(car2_start),
                "goal": list(car2_goal),
                "concepts": {"type": "Car", "priority": 2},
                "type": "Car",
                "priority": 2,
                "pos": list(car2_start),
                "layer_id": 4,
                "id": 2,
            },
        },
        "label_info": {
            "oracle_step": 40,
            "scenario": "two_car_symmetric_intersection_crossing",
        },
    }


def local_predicate_report(city):
    layerid2listid = {agent.layer_id: idx for idx, agent in enumerate(city.agents)}
    _, partial_agents, partial_world, partial_intersections, _ = city.local_planner.break_world_matrix(
        city.city_grid.clone(),
        city.agents,
        city.intersection_matrix.clone(),
        layerid2listid,
        rl_agent={3, 4},
    )
    for ego_name in ("Car_3", "Car_4"):
        pa = partial_agents[ego_name]
        pw = partial_world[ego_name]
        pi = partial_intersections[ego_name]
        ego_key = next(key for key in pa if key.startswith("ego_"))
        ego = pa[ego_key]
        ego_entity = f"Entity_{ego.type}_{ego.layer_id}"
        other_key = next(key for key in pa if not key.startswith("ego_") and not key.startswith("PH_"))
        other = pa[other_key]
        other_entity = f"Entity_{other.type}_{other.layer_id}"
        print(f"[{ego_name}]")
        print(f"  ego_entity={ego_entity}")
        print(f"  other_entity={other_entity}")
        print(f"  IsAtInter={predz3.IsAtInter(pw, pi, pa, ego_entity)}")
        print(f"  IsInInter={predz3.IsInInter(pw, pi, pa, ego_entity)}")
        print(f"  HigherPri={predz3.HigherPri(pw, pi, pa, ego_entity, other_entity)}")
        print(f"  CollidingClose={predz3.CollidingClose(pw, pi, pa, ego_entity, other_entity)}")


def find_symmetric_episode(simulation_config, max_offset):
    city, _ = CityLoader.from_yaml(**simulation_config)
    center_mask = get_center_intersection_mask(city)
    coords = center_mask.nonzero(as_tuple=False)
    top = int(coords[:, 0].min().item())
    bottom = int(coords[:, 0].max().item())
    left = int(coords[:, 1].min().item())
    right = int(coords[:, 1].max().item())
    street_mask = city.city_grid[STREET_ID] == TYPE_MAP["Traffic Street"]

    for offset in range(2, max_offset + 1):
        left_col = left - offset
        right_col = right + offset
        top_row = top - offset
        bottom_row = bottom + offset
        if min(left_col, top_row) < 0:
            continue
        if right_col >= street_mask.shape[1] or bottom_row >= street_mask.shape[0]:
            continue

        left_rows = [r for r in range(top - 2, bottom + 3) if bool(street_mask[r, left_col].item())]
        right_rows = [r for r in range(top - 2, bottom + 3) if bool(street_mask[r, right_col].item())]
        top_cols = [c for c in range(left - 2, right + 3) if bool(street_mask[top_row, c].item())]
        bottom_cols = [c for c in range(left - 2, right + 3) if bool(street_mask[bottom_row, c].item())]

        for row in left_rows:
            if row not in right_rows:
                continue
            for col in top_cols:
                if col not in bottom_cols:
                    continue
                episode = build_episode(
                    car1_start=(row, left_col),
                    car1_goal=(row, right_col),
                    car2_start=(top_row, col),
                    car2_goal=(bottom_row, col),
                )
                loaded_city, _ = CityLoader.from_yaml(**simulation_config, episode_cache=episode)
                car1, car2 = loaded_city.agents
                eta1 = first_intersection_index(car1, center_mask)
                eta2 = first_intersection_index(car2, center_mask)
                last1 = last_intersection_index(car1, center_mask)
                last2 = last_intersection_index(car2, center_mask)
                if None in (eta1, eta2, last1, last2):
                    continue
                if eta1 != eta2:
                    continue
                if eta1 <= 0 or eta2 <= 0:
                    continue
                enter_dir1 = step_direction(car1.global_traj, eta1 - 1, eta1)
                exit_dir1 = step_direction(car1.global_traj, last1, min(last1 + 1, len(car1.global_traj) - 1))
                enter_dir2 = step_direction(car2.global_traj, eta2 - 1, eta2)
                exit_dir2 = step_direction(car2.global_traj, last2, min(last2 + 1, len(car2.global_traj) - 1))
                if enter_dir1 != "right" or exit_dir1 != "right":
                    continue
                if enter_dir2 != "down" or exit_dir2 != "down":
                    continue
                return episode, loaded_city, {
                    "intersection_bbox": [top, left, bottom, right],
                    "eta": eta1,
                    "car1_start": [row, left_col],
                    "car1_goal": [row, right_col],
                    "car2_start": [top_row, col],
                    "car2_goal": [bottom_row, col],
                }
    raise RuntimeError("Failed to find a symmetric two-car crossing episode.")


def main():
    args = build_parser().parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    episode, city, meta = find_symmetric_episode(config["simulation"], args.max_offset)
    episode["city_grid"] = city.city_grid.clone()
    episode["label_info"].update(meta)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pkl.dump({0: episode}, f)
    print(f"Wrote episode to {output_path}")

    static_image_path = Path(args.static_image)
    static_image_path.parent.mkdir(parents=True, exist_ok=True)
    visualize_city(city, 4 * city.grid_size[0], -1, str(static_image_path))
    print(f"Wrote static image to {static_image_path}")
    print("Scenario:")
    print(episode["label_info"])
    print("Validation report:")
    local_predicate_report(city)


if __name__ == "__main__":
    main()
