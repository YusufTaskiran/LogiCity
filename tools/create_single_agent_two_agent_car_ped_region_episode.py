import os
import time
import yaml
import torch
import argparse
import importlib
import copy
import numpy as np
import pickle as pkl

from logicity.utils.logger import setup_logger
from logicity.utils.load import CityLoader
from logicity.utils.gym_wrapper import GymCityWrapper
from tools.create_two_agent_car_ped_region_episode import build_episode_cache


MIN_START_SEPARATION = 6.0


def parse_arguments():
    parser = argparse.ArgumentParser(description="Create single-agent cached episodes in the 4-agent two_agent_car_ped scene.")
    parser.add_argument("--log_dir", type=str, default="./log_rl")
    parser.add_argument("--exp", type=str, default="single_two_agent_car_ped_region")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--max_episodes", type=int, default=50)
    parser.add_argument("--require_stop", action="store_true")
    parser.add_argument("--save_worlds", action="store_true")
    parser.add_argument("--worlds_dir", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "--config",
        default="config/tasks/Nav/simple/experts/expert_episode_test_single_two_agent_car_ped.yaml",
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


def _agent_pos(episode_cache, agent_name):
    return np.asarray(episode_cache["agents"][agent_name]["pos"], dtype=np.float32)


def _passes_spawn_separation(episode_cache):
    agent_names = ["Car_1", "Car_2", "Car_3", "Pedestrian_1"]
    for idx, name_a in enumerate(agent_names):
        pos_a = _agent_pos(episode_cache, name_a)
        for name_b in agent_names[idx + 1 :]:
            pos_b = _agent_pos(episode_cache, name_b)
            if float(np.linalg.norm(pos_a - pos_b)) < MIN_START_SEPARATION:
                return False
    return True


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

    controlled_agent_names = ["Car_1", "Car_2"]

    while key < args.max_episodes:
        attempts += 1
        logger.info("Attempt %s | accepted %s/%s", attempts, key, args.max_episodes)
        base_city, _ = CityLoader.from_yaml(**simulation_config)
        episode_cache = build_episode_cache(base_city, controlled_agent_names, rng)
        if not _passes_spawn_separation(episode_cache):
            logger.info("Discarding attempt %s because spawned agents are too close together.", attempts)
            continue
        eval_env, cached_observation = make_env(simulation_config, episode_cache=episode_cache, return_cache=True)
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
            rew += reward
            cached_observation["Time_Obs"][step] = info

        if not info["is_success"]:
            logger.info("Discarding attempt %s because expert did not succeed for Car_1.", attempts)
            continue
        if args.require_stop and not stop_used:
            logger.info("Discarding attempt %s because no Stop action was used.", attempts)
            continue

        saved_episode = copy.deepcopy(episode_cache)
        saved_episode["city_grid"] = saved_episode["city_grid"].clone()
        saved_episode["label_info"] = {
            "action": 1 if stop_used else 0,
            "oracle_step": int(step),
            "stop_used": bool(stop_used),
            "assigned_priorities": episode_cache.get("label_info", {}).get("assigned_priorities", {}),
            "scene_type": "two_agent_car_ped",
        }
        all_episodes[key] = saved_episode
        rew_list.append(rew)
        success.append(1)
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

    output_path = args.output_path or os.path.join(args.log_dir, f"{args.exp}_episodes.pkl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info("Saving episodes to %s", output_path)
    with open(output_path, "wb") as f:
        pkl.dump(all_episodes, f)


if __name__ == "__main__":
    args = parse_arguments()
    logger = setup_logger(log_dir=args.log_dir, log_name=args.exp)
    logger.info("Collecting %s single-agent episodes in two_agent_car_ped scene from expert...", args.max_episodes)
    main(args, logger)
