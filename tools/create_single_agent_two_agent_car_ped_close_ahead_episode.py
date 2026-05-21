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
from logicity.utils.pred_converter.z3 import IsAhead, IsCloseAhead
from tools.create_two_agent_car_ped_region_episode import build_episode_cache


CLOSE_AHEAD_TEMPLATES = [
    {
        "name": "eastbound_lane_49",
        "car1_start": (25, 49),
        "car2_start": (28, 49),
        "goal": (77, 49),
    },
    {
        "name": "westbound_lane_53",
        "car1_start": (77, 53),
        "car2_start": (74, 53),
        "goal": (25, 53),
    },
    {
        "name": "southbound_lane_49",
        "car1_start": (49, 25),
        "car2_start": (49, 28),
        "goal": (49, 77),
    },
    {
        "name": "northbound_lane_53",
        "car1_start": (53, 77),
        "car2_start": (53, 74),
        "goal": (53, 25),
    },
]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Create single-agent cached close-ahead episodes in the 4-agent two_agent_car_ped scene.")
    parser.add_argument("--log_dir", type=str, default="./log_rl")
    parser.add_argument("--exp", type=str, default="single_two_agent_car_ped_close_ahead")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--max_episodes", type=int, default=50)
    parser.add_argument("--require_stop", action="store_true")
    parser.add_argument("--save_worlds", action="store_true")
    parser.add_argument("--worlds_dir", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "--config",
        default="config/tasks/Nav/thesis/experts/expert_episode_val.yaml",
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


def _set_agent(episode_cache, agent_name, start, goal):
    episode_cache["agents"][agent_name]["start"] = np.array(start, dtype=np.int64)
    episode_cache["agents"][agent_name]["goal"] = np.array(goal, dtype=np.int64)
    episode_cache["agents"][agent_name]["pos"] = np.array(start, dtype=np.int64)


def build_close_ahead_episode_cache(base_city, controlled_agent_names, rng):
    episode_cache = build_episode_cache(base_city, controlled_agent_names, rng)
    template = CLOSE_AHEAD_TEMPLATES[int(rng.integers(0, len(CLOSE_AHEAD_TEMPLATES)))]
    _set_agent(episode_cache, "Car_1", template["car1_start"], template["goal"])
    _set_agent(episode_cache, "Car_2", template["car2_start"], template["goal"])
    episode_cache.setdefault("label_info", {})
    episode_cache["label_info"]["scenario_type"] = "close_ahead"
    episode_cache["label_info"]["template"] = template["name"]
    return episode_cache


def find_agent(eval_env, name):
    for agent in eval_env.env.agents:
        if f"{agent.type}_{agent.id}" == name:
            return agent
    raise KeyError(f"Agent {name} not found in environment.")


def entity_name(agent):
    return f"Entity_{agent.type}_{agent.layer_id}"


def scenario_flags(eval_env):
    world_matrix = eval_env.env.city_grid
    intersect_matrix = eval_env.env.intersection_matrix
    agents = eval_env.env.agents
    ego_agent = find_agent(eval_env, "Car_1")
    other_agent = find_agent(eval_env, "Car_2")
    ego_entity = entity_name(ego_agent)
    other_entity = entity_name(other_agent)
    return {
        "ahead": bool(IsAhead(world_matrix, intersect_matrix, agents, ego_entity, other_entity)),
        "close_ahead": bool(IsCloseAhead(world_matrix, intersect_matrix, agents, ego_entity, other_entity)),
    }


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
        episode_cache = build_close_ahead_episode_cache(base_city, controlled_agent_names, rng)
        eval_env, cached_observation = make_env(simulation_config, episode_cache=episode_cache, return_cache=True)
        model = algorithm_class(eval_env)
        obs = eval_env.init()

        initial_flags = scenario_flags(eval_env)

        rew = 0.0
        step = 0
        done = False
        stop_used = False
        observed_close_ahead = bool(initial_flags["close_ahead"])
        start_t = time.time()

        while not done:
            step += 1
            action, _ = model.predict(obs, deterministic=True)
            if int(action) == 1:
                stop_used = True
            obs, reward, done, info = eval_env.step(action)
            rew += reward
            cached_observation["Time_Obs"][step] = info
            current_flags = scenario_flags(eval_env)
            observed_close_ahead = observed_close_ahead or bool(current_flags["close_ahead"])

        if not info["is_success"]:
            logger.info("Discarding attempt %s because expert did not succeed for Car_1.", attempts)
            continue
        if args.require_stop and not stop_used:
            logger.info("Discarding attempt %s because no Stop action was used.", attempts)
            continue
        if not observed_close_ahead:
            logger.info("Discarding attempt %s because close_ahead never occurred during rollout.", attempts)
            continue

        saved_episode = copy.deepcopy(episode_cache)
        saved_episode["city_grid"] = saved_episode["city_grid"].clone()
        saved_episode["label_info"] = {
            "action": 1 if stop_used else 0,
            "stop_used": bool(stop_used),
            "assigned_priorities": episode_cache.get("label_info", {}).get("assigned_priorities", {}),
            "scene_type": "two_agent_car_ped",
            "scenario_type": "close_ahead",
            "template": episode_cache.get("label_info", {}).get("template"),
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
    logger.info("Collecting %s single-agent close-ahead episodes in two_agent_car_ped scene from expert...", args.max_episodes)
    main(args, logger)
