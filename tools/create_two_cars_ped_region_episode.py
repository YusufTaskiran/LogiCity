import os
import time
import yaml
import torch
import argparse
import importlib
import numpy as np
import pickle as pkl

from logicity.utils.load import CityLoader
from logicity.utils.logger import setup_logger
from logicity.utils.gym_wrapper import GymCityWrapper


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
    parser = argparse.ArgumentParser(description="Create 2-car-1-ped center-intersection episodes.")
    parser.add_argument("--log_dir", type=str, default="./log_rl")
    parser.add_argument("--exp", type=str, default="two_cars_ped_region")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--max_episodes", type=int, default=10)
    parser.add_argument("--require_stop", action="store_true")
    parser.add_argument("--save_worlds", action="store_true")
    parser.add_argument("--worlds_dir", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "--config",
        default="config/tasks/Nav/simple/experts/expert_episode_val_two_cars_ped.yaml",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def dynamic_import(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def make_env(simulation_config, episode_cache=None, return_cache=False):
    city, cached_observation = CityLoader.from_yaml(**simulation_config, episode_cache=episode_cache)
    env = GymCityWrapper(city)
    if return_cache:
        return env, cached_observation
    return env


def build_episode_cache(base_env, rng):
    episode_cache = base_env.save_episode()
    # Assign distinct car approaches through the center intersection.
    start_sides = rng.choice(list(CAR_REGIONS_2X2.keys()), size=2, replace=False).tolist()
    goal_choices = []
    for start_side in start_sides:
        goal_side = str(rng.choice([side for side in CAR_REGIONS_2X2.keys() if side != start_side]))
        goal_choices.append(goal_side)

    car_names = ["Car_1", "Car_2"]
    for idx, car_name in enumerate(car_names):
        start = CAR_REGIONS_2X2[start_sides[idx]][int(rng.integers(0, len(CAR_REGIONS_2X2[start_sides[idx]])))]
        goal = CAR_REGIONS_2X2[goal_choices[idx]][int(rng.integers(0, len(CAR_REGIONS_2X2[goal_choices[idx]])))]
        episode_cache["agents"][car_name]["start"] = np.array(start, dtype=np.int64)
        episode_cache["agents"][car_name]["goal"] = np.array(goal, dtype=np.int64)
        episode_cache["agents"][car_name]["pos"] = np.array(start, dtype=np.int64)

    # Randomize car priorities so HigherPri varies.
    priorities = [1, 2]
    rng.shuffle(priorities)
    for idx, car_name in enumerate(car_names):
        episode_cache["agents"][car_name]["priority"] = int(priorities[idx])
        concepts = dict(episode_cache["agents"][car_name]["concepts"])
        concepts["priority"] = int(priorities[idx])
        episode_cache["agents"][car_name]["concepts"] = concepts

    # Force the pedestrian to take a diagonal crossing through the center.
    diagonal_pairs = [("nw", "se"), ("ne", "sw")]
    start_side, goal_side = diagonal_pairs[int(rng.integers(0, len(diagonal_pairs)))]
    if int(rng.integers(0, 2)) == 1:
        start_side, goal_side = goal_side, start_side
    ped_start = PED_REGIONS_2X2[start_side][int(rng.integers(0, len(PED_REGIONS_2X2[start_side])))]
    ped_goal = PED_REGIONS_2X2[goal_side][int(rng.integers(0, len(PED_REGIONS_2X2[goal_side])))]
    episode_cache["agents"]["Pedestrian_1"]["start"] = np.array(ped_start, dtype=np.int64)
    episode_cache["agents"]["Pedestrian_1"]["goal"] = np.array(ped_goal, dtype=np.int64)
    episode_cache["agents"]["Pedestrian_1"]["pos"] = np.array(ped_start, dtype=np.int64)
    return episode_cache


def main(args, logger):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    config = load_config(args.config)
    simulation_config = config["simulation"]
    rl_config = config["stable_baselines"]
    logger.info("Simulation config: %s", simulation_config)

    algorithm_class = dynamic_import("logicity.rl_agent.alg", rl_config["algorithm"])
    assert rl_config["algorithm"] == "ExpertCollector"

    all_episodes = {}
    rew_list = []
    success = []
    key = 0
    attempts = 0
    worlds_dir = args.worlds_dir or os.path.join(args.log_dir, f"{args.exp}_worlds")
    if args.save_worlds:
        os.makedirs(worlds_dir, exist_ok=True)

    while key < args.max_episodes:
        attempts += 1
        logger.info("Attempt %s | accepted %s/%s", attempts, key, args.max_episodes)
        base_env, _ = make_env(simulation_config, None, True)
        episode_cache = build_episode_cache(base_env, rng)
        eval_env, cached_observation = make_env(simulation_config, episode_cache, True)
        model = algorithm_class(eval_env)
        obs = eval_env.init()

        rew = 0
        step = 0
        done = False
        stop_used = False
        start_t = time.time()

        while not done:
            step += 1
            action, _ = model.predict(obs, deterministic=True)
            if int(action) == 1:
                stop_used = True
            obs, reward, done, info = eval_env.step(action)
            rew += reward
            cached_observation["Time_Obs"][step] = info

        if not info["is_success"]:
            logger.info("Discarding attempt %s because expert did not succeed.", attempts)
            continue
        if args.require_stop and not stop_used:
            logger.info("Discarding attempt %s because no Stop action was used.", attempts)
            continue

        episode_cache = eval_env.save_episode()
        episode_cache["label_info"] = {
            "action": 1 if stop_used else 0,
            "oracle_step": step,
            "stop_used": stop_used,
            "car_priorities": {
                "Car_1": int(episode_cache["agents"]["Car_1"]["priority"]),
                "Car_2": int(episode_cache["agents"]["Car_2"]["priority"]),
            },
        }
        all_episodes[key] = episode_cache
        rew_list.append(rew)
        success.append(1)
        if args.save_worlds:
            world_path = os.path.join(worlds_dir, f"{args.exp}_{key}.pkl")
            with open(world_path, "wb") as f:
                pkl.dump(cached_observation, f)
        logger.info("Episode %s took %s steps and %.3fs.", key, step, time.time() - start_t)
        logger.info("Episode %s score: %s", key, rew)
        logger.info("Episode %s label info: %s", key, episode_cache["label_info"])
        key += 1

    logger.info("Generated %s episodes after %s attempts.", len(all_episodes), attempts)
    logger.info("Success rate: %s", np.mean(success) if success else 0.0)
    logger.info("Mean score achieved: %s", np.mean(rew_list) if rew_list else 0.0)

    output_path = args.output_path or os.path.join(args.log_dir, f"{args.exp}_episodes.pkl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info("Saving episodes to %s", output_path)
    with open(output_path, "wb") as f:
        pkl.dump(all_episodes, f)


if __name__ == "__main__":
    args = parse_arguments()
    logger = setup_logger(log_dir=args.log_dir, log_name=args.exp)
    logger.info("Collecting %s region-based two_cars_ped episodes from expert...", args.max_episodes)
    main(args, logger)
