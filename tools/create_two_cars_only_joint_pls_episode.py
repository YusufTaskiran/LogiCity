import copy
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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create 2-car-only center-intersection episodes for joint-PLS training, including same-priority cases."
    )
    parser.add_argument("--log_dir", type=str, default="./log_rl")
    parser.add_argument("--exp", type=str, default="two_cars_only_joint_pls")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--max_episodes", type=int, default=50)
    parser.add_argument("--require_stop", action="store_true")
    parser.add_argument("--save_worlds", action="store_true")
    parser.add_argument("--worlds_dir", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "--same_priority_fraction",
        type=float,
        default=0.5,
        help="Fraction of accepted episodes that should use equal priorities for both RL cars.",
    )
    parser.add_argument(
        "--config",
        default="config/tasks/Nav/thesis/experts/expert_episode_test_two_cars_only.yaml",
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


def _rand_point(rng, region_points):
    return region_points[int(rng.integers(0, len(region_points)))]


def _assign_priorities(rng, same_priority: bool):
    if same_priority:
        priority = int(rng.choice([1, 2]))
        return [priority, priority]
    priorities = [1, 2]
    rng.shuffle(priorities)
    return priorities


def build_episode_cache(base_env, rng, same_priority: bool):
    episode_cache = base_env.save_episode()
    start_sides = rng.choice(list(CAR_REGIONS_2X2.keys()), size=2, replace=False).tolist()
    goal_choices = []
    for start_side in start_sides:
        goal_candidates = [side for side in CAR_REGIONS_2X2.keys() if side != start_side]
        goal_choices.append(str(rng.choice(goal_candidates)))

    priorities = _assign_priorities(rng, same_priority=same_priority)
    car_names = ["Car_1", "Car_2"]
    for idx, car_name in enumerate(car_names):
        start = _rand_point(rng, CAR_REGIONS_2X2[start_sides[idx]])
        goal = _rand_point(rng, CAR_REGIONS_2X2[goal_choices[idx]])
        episode_cache["agents"][car_name]["start"] = np.array(start, dtype=np.int64)
        episode_cache["agents"][car_name]["goal"] = np.array(goal, dtype=np.int64)
        episode_cache["agents"][car_name]["pos"] = np.array(start, dtype=np.int64)
        episode_cache["agents"][car_name]["priority"] = int(priorities[idx])
        concepts = dict(episode_cache["agents"][car_name]["concepts"])
        concepts["priority"] = int(priorities[idx])
        episode_cache["agents"][car_name]["concepts"] = concepts

    episode_cache.setdefault("label_info", {})
    episode_cache["label_info"]["scene_type"] = "two_cars_intersection"
    episode_cache["label_info"]["same_priority"] = bool(priorities[0] == priorities[1])
    episode_cache["label_info"]["car_priorities"] = {
        "Car_1": int(priorities[0]),
        "Car_2": int(priorities[1]),
    }
    episode_cache["label_info"]["start_sides"] = {
        "Car_1": start_sides[0],
        "Car_2": start_sides[1],
    }
    episode_cache["label_info"]["goal_sides"] = {
        "Car_1": goal_choices[0],
        "Car_2": goal_choices[1],
    }
    return episode_cache


def main(args, logger):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    config = load_config(args.config)
    simulation_config = config["simulation"]
    rl_config = config["stable_baselines"]

    algorithm_class = dynamic_import("logicity.rl_agent.alg", rl_config["algorithm"])
    assert rl_config["algorithm"] == "ExpertCollector"

    all_episodes = {}
    rew_list = []
    success = []
    key = 0
    attempts = 0
    same_priority_target = int(round(float(np.clip(args.same_priority_fraction, 0.0, 1.0)) * args.max_episodes))
    same_priority_count = 0
    worlds_dir = args.worlds_dir or os.path.join(args.log_dir, f"{args.exp}_worlds")
    if args.save_worlds:
        os.makedirs(worlds_dir, exist_ok=True)

    while key < args.max_episodes:
        attempts += 1
        need_same_priority = same_priority_count < same_priority_target
        remaining_slots = args.max_episodes - key
        remaining_same_priority_needed = same_priority_target - same_priority_count
        if remaining_same_priority_needed >= remaining_slots:
            same_priority = True
        elif need_same_priority:
            same_priority = bool(rng.random() < 0.5)
        else:
            same_priority = False

        logger.info(
            "Attempt %s | accepted %s/%s | same_priority_target=%s | same_priority_count=%s | sampling_same_priority=%s",
            attempts,
            key,
            args.max_episodes,
            same_priority_target,
            same_priority_count,
            same_priority,
        )
        base_env, _ = make_env(simulation_config, None, True)
        episode_cache = build_episode_cache(base_env, rng, same_priority=same_priority)
        eval_env, cached_observation = make_env(simulation_config, episode_cache, True)
        model = algorithm_class(eval_env)
        obs = eval_env.init()

        rew = 0.0
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
            rew += float(reward)
            cached_observation["Time_Obs"][step] = info

        if not info["is_success"]:
            logger.info("Discarding attempt %s because expert did not succeed.", attempts)
            continue
        if args.require_stop and not stop_used:
            logger.info("Discarding attempt %s because no Stop action was used.", attempts)
            continue

        saved_episode = eval_env.save_episode()
        saved_episode["label_info"] = copy.deepcopy(episode_cache["label_info"])
        saved_episode["label_info"]["action"] = 1 if stop_used else 0
        saved_episode["label_info"]["oracle_step"] = int(step)
        saved_episode["label_info"]["stop_used"] = bool(stop_used)
        saved_episode["label_info"]["equal_priority_edge_case"] = bool(saved_episode["label_info"]["same_priority"])
        all_episodes[key] = saved_episode
        rew_list.append(rew)
        success.append(1)
        if saved_episode["label_info"]["same_priority"]:
            same_priority_count += 1
        if args.save_worlds:
            world_path = os.path.join(worlds_dir, f"{args.exp}_{key}.pkl")
            with open(world_path, "wb") as f:
                pkl.dump(cached_observation, f)
        logger.info("Episode %s took %s steps and %.3fs.", key, step, time.time() - start_t)
        logger.info("Episode %s score: %s", key, rew)
        logger.info("Episode %s label info: %s", key, saved_episode["label_info"])
        key += 1

    logger.info("Generated %s episodes after %s attempts.", len(all_episodes), attempts)
    logger.info("Success rate: %s", np.mean(success) if success else 0.0)
    logger.info("Mean score achieved: %s", np.mean(rew_list) if rew_list else 0.0)
    logger.info("Accepted same-priority episodes: %s/%s", same_priority_count, len(all_episodes))

    output_path = args.output_path or os.path.join(args.log_dir, f"{args.exp}_episodes.pkl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info("Saving episodes to %s", output_path)
    with open(output_path, "wb") as f:
        pkl.dump(all_episodes, f)


if __name__ == "__main__":
    args = parse_arguments()
    logger = setup_logger(log_dir=args.log_dir, log_name=args.exp)
    logger.info(
        "Collecting %s two-car intersection episodes for joint PLS, with same_priority_fraction=%s...",
        args.max_episodes,
        args.same_priority_fraction,
    )
    main(args, logger)
