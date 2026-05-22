import os
import sys
import time
import yaml
import torch
import random as pyrandom
import argparse
import importlib
import numpy as np
import pickle as pkl
import multiprocessing as mp
from tqdm import trange
from logicity.core.config import *
from logicity.utils.load import CityLoader
from logicity.utils.logger import setup_logger
from logicity.utils.vis import visualize_city
# RL
from logicity.rl_agent.alg import *
from logicity.utils.gym_wrapper import GymCityWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv
from logicity.utils.gym_callback import EvalCheckpointCallback

def parse_arguments():
    parser = argparse.ArgumentParser(description='Logic-based city simulation.')
    # logger
    parser.add_argument('--log_dir', type=str, default="./log_rl")
    parser.add_argument('--exp', type=str, default="transfer_easy_val")
    parser.add_argument('--replace_key', type=list, default=[])
    # seed
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--max_episodes', type=int, default=40)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--vis_count', type=int, default=0)
    parser.add_argument('--progress_every', type=int, default=5)
    # RL
    parser.add_argument('--config', default='config/tasks/Nav/transfer/easy/expert_episode_val.yaml', help='Configure file for this RL exp.')

    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def dynamic_import(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def make_env(simulation_config, return_cache=False): 
    # Unpack arguments from simulation_config and pass them to CityLoader
    city, cached_observation = CityLoader.from_yaml(**simulation_config)
    env = GymCityWrapper(city)
    if return_cache: 
        return env, cached_observation
    else:
        return env
    
def make_envs(simulation_config, rank):
    """
    Utility function for multiprocessed env.
    
    :param simulation_config: The configuration for the simulation.
    :param rank: Unique index for each environment to ensure different seeds.
    :return: A function that creates a single environment.
    """
    def _init():
        env = make_env(simulation_config)
        env.seed(rank + 1000)  # Optional: set a unique seed for each environment
        return env
    return _init


def _get_center_intersection_mask(eval_env, selector="center"):
    intersection_blocks = eval_env.env.intersection_matrix[2]
    labels = torch.unique(intersection_blocks)
    labels = labels[labels > 0]
    if labels.numel() == 0:
        return None

    if selector != "center":
        raise ValueError("Unsupported intersection selector: {}".format(selector))

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


def _agent_route_crosses_mask(agent, mask):
    traj = agent.global_traj
    if traj is None or len(traj) == 0:
        return False
    coords = traj.long()
    return bool(mask[coords[:, 0], coords[:, 1]].any().item())


def _first_intersection_index(agent, mask):
    traj = agent.global_traj
    if traj is None or len(traj) == 0:
        return None
    coords = traj.long()
    hits = mask[coords[:, 0], coords[:, 1]].nonzero(as_tuple=False)
    if hits.numel() == 0:
        return None
    return int(hits[0].item())


def _traj_direction_at_index(agent, traj_idx):
    traj = agent.global_traj
    if traj is None or len(traj) < 2 or traj_idx is None:
        return None
    if traj_idx <= 0:
        delta = traj[1] - traj[0]
    else:
        delta = traj[traj_idx] - traj[traj_idx - 1]
    dx = int(delta[0].item())
    dy = int(delta[1].item())
    if abs(dx) >= abs(dy):
        return "down" if dx > 0 else "up"
    return "right" if dy > 0 else "left"


def _passes_episode_prefilter(eval_env, episode_generation):
    mode = episode_generation.get("mode", "legacy_balanced")
    if mode not in ("intersection_biased_success_only", "intersection_biased_legacy_balanced"):
        return True, {}

    selector = episode_generation.get("intersection_selector", "center")
    require_ego = episode_generation.get("require_ego_intersection", True)
    require_all = episode_generation.get("require_all_agents_intersection", False)
    min_other = episode_generation.get("min_other_intersection_agents", 1)
    min_close_other = episode_generation.get("min_temporally_close_other_agents", 1)
    max_other_eta_gap = episode_generation.get("max_other_intersection_eta_gap", None)
    require_ped_overlap = episode_generation.get("require_pedestrian_temporal_overlap", False)
    max_ped_eta_gap = episode_generation.get("max_pedestrian_intersection_eta_gap", None)
    require_conflicting_approach = episode_generation.get("require_conflicting_approach", False)

    mask = _get_center_intersection_mask(eval_env, selector=selector)
    if mask is None:
        return False, {"reason": "no_intersection_mask"}

    ego_layer_id = eval_env.agent_layer_id
    ego_crosses = False
    other_crosses = 0
    total_crosses = 0
    ego_eta = None
    ego_direction = None
    other_car_etas = []
    pedestrian_etas = []
    conflicting_approach = False
    for agent in eval_env.env.agents:
        crosses = _agent_route_crosses_mask(agent, mask)
        eta = _first_intersection_index(agent, mask) if crosses else None
        direction = _traj_direction_at_index(agent, eta) if crosses else None
        if crosses:
            total_crosses += 1
        if agent.layer_id == ego_layer_id:
            ego_crosses = crosses
            ego_eta = eta
            ego_direction = direction
        elif crosses:
            other_crosses += 1
            if agent.type == "Car":
                other_car_etas.append(eta)
            elif agent.type == "Pedestrian":
                pedestrian_etas.append(eta)
            if require_conflicting_approach and ego_direction is not None and direction is not None and direction != ego_direction:
                conflicting_approach = True
        if require_all and not crosses:
            return False, {
                "reason": "agent_misses_intersection",
                "ego_crosses": ego_crosses,
                "other_crosses": other_crosses,
                "total_crosses": total_crosses,
            }

    if require_ego and not ego_crosses:
        return False, {
            "reason": "ego_misses_intersection",
            "ego_crosses": ego_crosses,
            "other_crosses": other_crosses,
            "total_crosses": total_crosses,
        }
    if other_crosses < min_other:
        return False, {
            "reason": "too_few_other_agents_cross",
            "ego_crosses": ego_crosses,
            "other_crosses": other_crosses,
            "total_crosses": total_crosses,
        }

    close_other_count = 0
    if ego_eta is not None and max_other_eta_gap is not None:
        close_other_count = sum(1 for eta in other_car_etas if abs(eta - ego_eta) <= max_other_eta_gap)
        if close_other_count < min_close_other:
            return False, {
                "reason": "too_few_temporally_close_other_agents",
                "ego_crosses": ego_crosses,
                "ego_eta": ego_eta,
                "other_crosses": other_crosses,
                "close_other_count": close_other_count,
                "total_crosses": total_crosses,
            }

    pedestrian_close = None
    if require_ped_overlap:
        if ego_eta is None or max_ped_eta_gap is None or len(pedestrian_etas) == 0:
            return False, {
                "reason": "missing_pedestrian_temporal_overlap",
                "ego_crosses": ego_crosses,
                "ego_eta": ego_eta,
                "other_crosses": other_crosses,
                "total_crosses": total_crosses,
            }
        pedestrian_close = any(abs(eta - ego_eta) <= max_ped_eta_gap for eta in pedestrian_etas)
        if not pedestrian_close:
            return False, {
                "reason": "pedestrian_eta_too_far",
                "ego_crosses": ego_crosses,
                "ego_eta": ego_eta,
                "other_crosses": other_crosses,
                "total_crosses": total_crosses,
            }

    if require_conflicting_approach and not conflicting_approach:
        return False, {
            "reason": "no_conflicting_approach",
            "ego_crosses": ego_crosses,
            "ego_eta": ego_eta,
            "other_crosses": other_crosses,
            "total_crosses": total_crosses,
        }

    return True, {
        "ego_crosses": ego_crosses,
        "ego_eta": ego_eta,
        "ego_direction": ego_direction,
        "other_crosses": other_crosses,
        "close_other_count": close_other_count,
        "pedestrian_close": pedestrian_close,
        "conflicting_approach": conflicting_approach,
        "total_crosses": total_crosses,
    }

def _generate_success_only_batch(worker_id, config_path, seed, target_episodes, vis_count=0, progress_every=5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    pyrandom.seed(seed)

    config = load_config(config_path)
    simulation_config = config["simulation"]
    rl_config = config["stable_baselines"]
    episode_generation = config.get("episode_generation", {})
    algorithm_class = dynamic_import("logicity.rl_agent.alg", rl_config["algorithm"])
    assert rl_config["algorithm"] == "ExpertCollector"

    episodes = []
    rewards = []
    worlds = []
    attempts = 0
    episodes_with_stop = 0

    while len(episodes) < target_episodes:
        attempts += 1
        eval_env, cached_observation = make_env(simulation_config, True)
        model = algorithm_class(eval_env)
        o, tem_episode = eval_env.reset(True)
        prefilter_ok, prefilter_info = _passes_episode_prefilter(eval_env, episode_generation)
        if not prefilter_ok:
            continue
        rew = 0
        step = 0
        done = False
        stop_count = 0
        capture_world = len(worlds) < vis_count

        while not done:
            step += 1
            action, _ = model.predict(o, deterministic=True)
            if int(action) == 3:
                stop_count += 1
            o, r, done, info = eval_env.step(action)
            if capture_world:
                cached_observation["Time_Obs"][step] = info
            rew += r

        success_flag = bool(info.get("success", info.get("is_success", False)))
        if not success_flag:
            continue

        tem_episode["label_info"] = {
            "oracle_step": step,
            "stop_count": stop_count,
            "has_stop": stop_count > 0,
        }
        tem_episode["label_info"].update(prefilter_info)
        episodes.append(tem_episode)
        rewards.append(rew)
        if stop_count > 0:
            episodes_with_stop += 1
        if capture_world:
            worlds.append(cached_observation)
        if progress_every > 0 and (len(episodes) % progress_every == 0 or len(episodes) == target_episodes):
            print(
                "[worker {}] accepted {}/{} episodes after {} attempts".format(
                    worker_id, len(episodes), target_episodes, attempts
                ),
                flush=True,
            )

    return {
        "worker_id": worker_id,
        "episodes": episodes,
        "rewards": rewards,
        "worlds": worlds,
        "attempts": attempts,
        "episodes_with_stop": episodes_with_stop,
    }

def _run_parallel_success_only(args, logger):
    counts = [args.max_episodes // args.num_workers] * args.num_workers
    for idx in range(args.max_episodes % args.num_workers):
        counts[idx] += 1

    vis_budget = [0] * args.num_workers
    for idx in range(min(args.vis_count, args.max_episodes)):
        vis_budget[idx % args.num_workers] += 1

    logger.info("Generating {} successful episodes with {} workers.".format(args.max_episodes, args.num_workers))
    ctx = mp.get_context("spawn")
    worker_args = [
        (
            worker_id,
            args.config,
            args.seed + worker_id * 1000,
            counts[worker_id],
            vis_budget[worker_id],
            args.progress_every,
        )
        for worker_id in range(args.num_workers)
        if counts[worker_id] > 0
    ]

    results = []
    with ctx.Pool(processes=len(worker_args)) as pool:
        results = pool.starmap(_generate_success_only_batch, worker_args)

    all_episodes = {}
    worlds = []
    rew_list = []
    key = 0
    total_attempts = 0
    total_stop_episodes = 0
    for result in sorted(results, key=lambda item: item["worker_id"]):
        total_attempts += result["attempts"]
        rew_list.extend(result["rewards"])
        worlds.extend(result["worlds"])
        total_stop_episodes += result.get("episodes_with_stop", 0)
        for episode in result["episodes"]:
            all_episodes[key] = episode
            key += 1

    assert len(all_episodes) == args.max_episodes
    mean_reward = float(np.mean(rew_list)) if rew_list else 0.0
    logger.info("Success rate: 1.0")
    logger.info("Mean Score achieved: {}".format(mean_reward))
    logger.info("Total attempts across workers: {}".format(total_attempts))
    logger.info("Acceptance ratio: {:.4f}".format(args.max_episodes / total_attempts if total_attempts > 0 else 0.0))
    logger.info("Accepted episodes with at least one Stop: {}/{}".format(total_stop_episodes, args.max_episodes))

    for ts in range(min(len(worlds), 5)):
        with open(os.path.join(args.log_dir, "{}_{}.pkl".format(args.exp, ts)), "wb") as f:
            pkl.dump(worlds[ts], f)

    with open(os.path.join(args.log_dir, "{}_episodes.pkl".format(args.exp)), "wb") as f:
        pkl.dump(all_episodes, f)

def main(args, logger):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pyrandom.seed(args.seed)
    config = load_config(args.config)
    # simulation config
    simulation_config = config["simulation"]
    logger.info("Simulation config: {}".format(simulation_config))
    # RL config
    rl_config = config['stable_baselines']
    logger.info("RL config: {}".format(rl_config))
    episode_generation = config.get("episode_generation", {})
    generation_mode = episode_generation.get("mode", "legacy_balanced")

    if args.num_workers > 1:
        if generation_mode not in ("success_only", "intersection_biased_success_only"):
            raise ValueError("Parallel episode generation currently supports only success_only-style modes.")
        _run_parallel_success_only(args, logger)
        return

    # Dynamic import of the RL algorithm
    algorithm_class = dynamic_import(
        "logicity.rl_agent.alg",  # Adjust the module path as needed
        rl_config["algorithm"]
    )
    
    # Checkpoint evaluation
    rew_list = []
    worlds = []
    success = []
    all_episodes = {}
    stop_episode_count = 0
    key = 0
    vis_id = list(range(min(args.vis_count, args.max_episodes)))
    # 0: Slow, 1: Normal, 2: Fast, 3: Stop
    # Test
    desired_num = 25 if 'test' in args.config else 10
    num_desired = {}
    num_counter = {}
    if generation_mode == "legacy_balanced":
        num_desired = {
            'ambulance':{
                3: desired_num
            },
            'bus':{
                3: desired_num
            },
            'police':{
                3: desired_num
            },
            'normal':{
                3: desired_num
            }
        }
        num_counter = {
            'ambulance':{
                3: 0
            },
            'bus':{
                3: 0
            },
            'police':{
                3: 0
            },
            'normal':{
                3: 0
            }
        }
    # val
    # num_desired = {
    #     'police':{
    #         0: 10,
    #         2: 10,
    #         3: 5
    #     },
    #     'ambulance':{
    #         0: 5,
    #         3: 10
    #     },
    #     'reckless':{
    #         2: 5,
    #         3: 10
    #     },
    #     'bus':{
    #         2: 5,
    #         3: 10
    #     },
    #     'tiro': {
    #         0: 10,
    #         3: 10
    #     },
    #     'normal':{
    #         3: 10
    #     }
    # }
    # num_counter = {
    #     'police':{
    #         0: 0,
    #         2: 0,
    #         3: 0
    #     },
    #     'ambulance':{
    #         0: 0,
    #         3: 0
    #     },
    #     'reckless':{
    #         2: 0,
    #         3: 0
    #     },
    #     'bus':{
    #         2: 0,
    #         3: 0
    #     },
    #     'tiro': {
    #         0: 0,
    #         3: 0
    #     },
    #     'normal':{
    #         3: 0
    #     },
    #     'bus':{
    #         3: 0
    #     }
    # }
    while key < args.max_episodes: 
        if generation_mode == "legacy_balanced":
            # print current counter and desired in a table
            logger.info("Current counter and desired in a table:")
            logger.info("Concept | Speed | Counter | Desired")
            for concept in num_desired:
                for speed in num_desired[concept]:
                    logger.info("{} | {} | {} | {}".format(concept, speed, num_counter[concept][speed], num_desired[concept][speed]))
        logger.info("Trying to creat episode {} ...".format(key))
        eval_env, cached_observation = make_env(simulation_config, True)
        assert rl_config["algorithm"] == "ExpertCollector"
        model = algorithm_class(eval_env)
        o, tem_episodes = eval_env.reset(True)
        prefilter_ok, prefilter_info = _passes_episode_prefilter(eval_env, episode_generation)
        if not prefilter_ok:
            continue
        concept = 'normal'
        if generation_mode in ("legacy_balanced", "intersection_biased_legacy_balanced"):
            # Legacy balanced generation depends on the RL car concept label.
            cached_observation["Static Info"]["Agents"]["Car_3"]['concepts'] = tem_episodes['agents']['Car_1']['concepts']
            for current_concept in num_desired:
                if current_concept in tem_episodes['agents']['Car_1']['concepts']:
                    concept = current_concept
                    break
        if generation_mode in ("legacy_balanced", "intersection_biased_legacy_balanced"):
            skip = False
            using_dict = num_counter[concept]
            checking_dict = num_desired[concept]
            for speed in num_desired[concept]:
                if num_counter[concept][speed] < num_desired[concept][speed]:
                    skip = False
                    break
                logger.info("Skipping episode due to counter")
                skip = True
            if skip:
                continue
        else:
            using_dict = {}
            checking_dict = {}
        rew = 0    
        step = 0   
        d = False
        save = generation_mode in ("success_only", "intersection_biased_success_only")
        label_action = None
        stop_count = 0
        s = time.time()
        while not d:
            step += 1
            action, _ = model.predict(o, deterministic=True)
            if int(action) == 3:
                stop_count += 1
            o, r, d, i = eval_env.step(action)
            if key in vis_id:
                cached_observation["Time_Obs"][step] = i
            if generation_mode in ("legacy_balanced", "intersection_biased_legacy_balanced"):
                action = model.predict(o)[0]
                if (action in using_dict.keys()) and not save:
                    if using_dict[action] < checking_dict[action]:
                        label_action = action
                        save = True
            rew += r
        success_flag = bool(i.get("success", i.get("is_success", False)))
        if save and success_flag:
            logger.info("Episode {} took {} steps.".format(key, step))
            label_info = {
                'oracle_step': step,
                'stop_count': stop_count,
                'has_stop': stop_count > 0,
            }
            label_info.update(prefilter_info)
            if generation_mode in ("legacy_balanced", "intersection_biased_legacy_balanced"):
                label_info['concept'] = concept
            if label_action is not None:
                label_info['action'] = label_action
            logger.info("Episode {} took {} seconds.".format(key, time.time()-s))
            tem_episodes['label_info'] = label_info
            all_episodes[key] = tem_episodes
            rew_list.append(rew)
            success.append(1)
            if stop_count > 0:
                stop_episode_count += 1
            if key in vis_id:
                worlds.append(cached_observation)
            logger.info("Episode {} achieved a score of {}".format(key, rew))
            logger.info("Episode {} has label info: {}".format(key, label_info))
            key += 1
            if generation_mode in ("legacy_balanced", "intersection_biased_legacy_balanced") and label_action is not None:
                using_dict[label_action] += 1
    assert len(rew_list) == len(success) == args.max_episodes
    mean_reward = np.mean(rew_list)
    logger.info("Success rate: {}".format(np.mean(success)))
    logger.info("Mean Score achieved: {}".format(mean_reward))
    logger.info("Accepted episodes with at least one Stop: {}/{}".format(stop_episode_count, args.max_episodes))
    for ts in range(len(worlds)):
        with open(os.path.join(args.log_dir, "{}_{}.pkl".format(args.exp, ts)), "wb") as f:
            pkl.dump(worlds[ts], f)

    with open(os.path.join(args.log_dir, "{}_episodes.pkl".format(args.exp)), "wb") as f:
        pkl.dump(all_episodes, f)

if __name__ == '__main__':
    args = parse_arguments()
    logger = setup_logger(log_dir=args.log_dir, log_name=args.exp)
    # Sim mode, will use the logic-based simulator to run a simulation (no learning)
    logger.info("Collecting {} episodes from expert...".format(args.max_episodes))
    logger.info("Loading simulation config from {}.".format(args.config))
    e = time.time()
    main(args, logger)
    logger.info("Total time spent: {}".format(time.time()-e))
