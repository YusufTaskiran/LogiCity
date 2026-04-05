import os
import time
import yaml
import torch
import argparse
import numpy as np
import pickle as pkl

from logicity.utils.logger import setup_logger
from tools.run_shared_policy_two_agent import build_city, SharedPolicyTwoAgentRunner


CAR_REGIONS_2X2 = {
    "north": [(49, 25), (53, 25)],
    "south": [(49, 77), (53, 77)],
    "west": [(25, 49), (25, 53)],
    "east": [(77, 49), (77, 53)],
}

PED_REGIONS_2X2 = {
    "nw": [(41, 41), (41, 45), (45, 41)],
    "ne": [(41, 57), (41, 53), (45, 57)],
    "sw": [(57, 41), (53, 41), (57, 45)],
    "se": [(57, 57), (53, 57), (57, 53)],
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Create fixed 2-RL-car + rule-car + pedestrian intersection episodes.")
    parser.add_argument("--log_dir", type=str, default="./log_rl")
    parser.add_argument("--exp", type=str, default="two_agent_car_ped_region")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--max_episodes", type=int, default=50)
    parser.add_argument("--save_worlds", action="store_true")
    parser.add_argument("--worlds_dir", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "--config",
        default="config/tasks/Nav/simple/experts/expert_episode_test_two_agent_car_ped.yaml",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def _rand_point(rng, region_points):
    return region_points[int(rng.integers(0, len(region_points)))]


def build_episode_cache(base_city, controlled_agent_names, rng):
    episode_cache = {
        "city_grid": base_city.city_grid.clone(),
        "agents": {},
    }
    for agent in base_city.agents:
        name = f"{agent.type}_{agent.id}"
        episode_cache["agents"][name] = {
            "start": agent.start.clone().numpy(),
            "goal": agent.goal.clone().numpy(),
            "concepts": dict(agent.concepts),
            "type": agent.type,
            "priority": int(agent.priority),
            "pos": agent.pos.clone().numpy(),
            "layer_id": int(agent.layer_id),
            "id": int(agent.id),
        }

    car_names = ["Car_1", "Car_2", "Car_3"]
    start_sides = rng.choice(list(CAR_REGIONS_2X2.keys()), size=3, replace=False).tolist()
    for idx, car_name in enumerate(car_names):
        goal_candidates = [side for side in CAR_REGIONS_2X2.keys() if side != start_sides[idx]]
        goal_side = str(rng.choice(goal_candidates))
        start = _rand_point(rng, CAR_REGIONS_2X2[start_sides[idx]])
        goal = _rand_point(rng, CAR_REGIONS_2X2[goal_side])
        episode_cache["agents"][car_name]["start"] = np.array(start, dtype=np.int64)
        episode_cache["agents"][car_name]["goal"] = np.array(goal, dtype=np.int64)
        episode_cache["agents"][car_name]["pos"] = np.array(start, dtype=np.int64)

    priorities = [1, 2, 3]
    rng.shuffle(priorities)
    for idx, car_name in enumerate(car_names):
        episode_cache["agents"][car_name]["priority"] = int(priorities[idx])
        concepts = dict(episode_cache["agents"][car_name]["concepts"])
        concepts["priority"] = int(priorities[idx])
        episode_cache["agents"][car_name]["concepts"] = concepts

    diagonal_pairs = [("nw", "se"), ("ne", "sw")]
    start_side, goal_side = diagonal_pairs[int(rng.integers(0, len(diagonal_pairs)))]
    if int(rng.integers(0, 2)) == 1:
        start_side, goal_side = goal_side, start_side
    ped_start = _rand_point(rng, PED_REGIONS_2X2[start_side])
    ped_goal = _rand_point(rng, PED_REGIONS_2X2[goal_side])
    episode_cache["agents"]["Pedestrian_1"]["start"] = np.array(ped_start, dtype=np.int64)
    episode_cache["agents"]["Pedestrian_1"]["goal"] = np.array(ped_goal, dtype=np.int64)
    episode_cache["agents"]["Pedestrian_1"]["pos"] = np.array(ped_start, dtype=np.int64)

    episode_cache.setdefault("label_info", {})
    episode_cache["label_info"]["assigned_priorities"] = {
        name: int(episode_cache["agents"][name]["priority"]) for name in car_names
    }
    episode_cache["label_info"]["controlled_agents"] = list(controlled_agent_names)
    return episode_cache


def _passes_quality_filter(runner):
    return runner.passes_quality_filter()


def rollout_expert_episode(runner, cached_observation, controlled_agent_names):
    views = runner.collect_views()
    cached_observation["Time_Obs"][0] = {"World": runner.city.city_grid.clone()}
    step = 0
    done = False
    local_expert_stop = {name: 0 for name in controlled_agent_names}
    local_expert_hist = {name: {0: 0, 1: 0} for name in controlled_agent_names}

    while (not done) and (step < runner.horizon):
        step += 1
        actions = {}
        for agent in runner.controlled_agents:
            agent_name = runner.layer_id_to_name[agent.layer_id]
            expert_action = views[agent.layer_id]["expert_action"]
            if expert_action is None:
                expert_action = 0
            local_expert_hist[agent_name][int(expert_action)] = local_expert_hist[agent_name].get(int(expert_action), 0) + 1
            if int(expert_action) == 1:
                local_expert_stop[agent_name] += 1
            actions[agent.layer_id] = int(expert_action)
        views, _, done, info = runner.step(views, actions)
        cached_observation["Time_Obs"][step] = {"World": runner.city.city_grid.clone()}

    return {
        "steps": int(step),
        "done": bool(done),
        "success": bool(info["is_success"]),
        "overtime": bool(info["overtime"]),
        "any_fail": bool(info["any_fail"]),
        "agent_success": dict(info["agent_success"]),
        "expert_stop_count": local_expert_stop,
        "expert_action_hist": local_expert_hist,
    }


def main(args, logger):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    config = load_config(args.config)
    simulation_config = config["simulation"]
    controlled_agent_names = list(simulation_config["rl_agent"]["agent_names"])
    logger.info("Simulation config: %s", simulation_config)

    all_episodes = {}
    key = 0
    attempts = 0
    worlds_dir = args.worlds_dir or os.path.join(args.log_dir, f"{args.exp}_worlds")
    if args.save_worlds:
        os.makedirs(worlds_dir, exist_ok=True)

    while key < args.max_episodes:
        attempts += 1
        logger.info("Attempt %s | accepted %s/%s", attempts, key, args.max_episodes)
        base_city, _ = build_city(simulation_config)
        episode_cache = build_episode_cache(base_city, controlled_agent_names, rng)
        city, cached_observation = build_city(simulation_config, episode_cache=episode_cache)
        runner = SharedPolicyTwoAgentRunner(city, simulation_config["rl_agent"], controlled_agent_names)
        if not _passes_quality_filter(runner):
            logger.info("Discarding attempt %s because the constrained scene failed the route-quality filter.", attempts)
            continue

        start_t = time.time()
        rollout = rollout_expert_episode(runner, cached_observation, controlled_agent_names)
        if (not rollout["success"]) or (not all(bool(rollout["agent_success"].get(name, False)) for name in controlled_agent_names)):
            logger.info("Discarding attempt %s because expert did not finish both RL agents successfully.", attempts)
            continue

        saved_episode = {
            "city_grid": episode_cache["city_grid"].clone(),
            "agents": episode_cache["agents"],
            "label_info": {
                "oracle_step": int(rollout["steps"]),
                "controlled_agents": controlled_agent_names,
                "assigned_priorities": episode_cache["label_info"]["assigned_priorities"],
                "expert_stop_count": rollout["expert_stop_count"],
            },
        }
        all_episodes[key] = saved_episode
        if args.save_worlds:
            world_path = os.path.join(worlds_dir, f"{args.exp}_{key}.pkl")
            with open(world_path, "wb") as f:
                pkl.dump(cached_observation, f)
        logger.info(
            "Accepted episode %s in %.3fs | oracle_step=%s | assigned_priorities=%s | expert_stop_count=%s",
            key,
            time.time() - start_t,
            rollout["steps"],
            saved_episode["label_info"]["assigned_priorities"],
            rollout["expert_stop_count"],
        )
        key += 1

    if key < args.max_episodes:
        logger.warning("Only generated %s episodes after %s attempts.", key, attempts)

    output_path = args.output_path or os.path.join(args.log_dir, f"{args.exp}_episodes.pkl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info("Saving episodes to %s", output_path)
    with open(output_path, "wb") as f:
        pkl.dump(all_episodes, f)


if __name__ == "__main__":
    args = parse_arguments()
    logger = setup_logger(log_dir=args.log_dir, log_name=args.exp)
    logger.info("Collecting %s fixed two-agent-car-ped episodes from expert...", args.max_episodes)
    main(args, logger)
