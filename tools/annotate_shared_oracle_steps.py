import argparse
import copy
import os
import pickle as pkl

import numpy as np
import torch
import yaml

from logicity.utils.logger import setup_logger
from tools.run_shared_policy_two_agent import build_runner_from_episode


def parse_arguments():
    parser = argparse.ArgumentParser(description="Annotate cached episodes with shared multi-agent oracle steps.")
    parser.add_argument("--config", required=True, help="Shared-policy evaluation config.")
    parser.add_argument("--input_path", required=True, help="Input cached episode dataset (.pkl).")
    parser.add_argument("--output_path", required=True, help="Output dataset with joint_oracle_step labels.")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--log_dir", type=str, default="./log_rl")
    parser.add_argument("--exp", type=str, default="annotate_shared_oracle_steps")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def run_expert_shared_episode(simulation_config, episode_cache):
    _, _, runner = build_runner_from_episode(simulation_config, episode_cache)
    views = runner.collect_views()
    step = 0
    done = False
    max_steps = int(runner.horizon)
    info = {"is_success": False, "overtime": False, "any_fail": False}

    while (not done) and (step < max_steps):
        step += 1
        actions = {}
        for agent in runner.controlled_agents:
            if agent.layer_id in runner.parked_controlled_layers:
                continue
            expert_action = views[agent.layer_id]["expert_action"]
            if expert_action is None:
                expert_action = runner.stop_action_id
            actions[agent.layer_id] = int(expert_action)
        views, _, done, info = runner.step(views, actions)

    return {
        "joint_oracle_step": int(step),
        "joint_oracle_success": bool(info.get("is_success", False)),
        "joint_oracle_overtime": bool(info.get("overtime", False)),
        "joint_oracle_fail": bool(info.get("any_fail", False)),
    }


def main(args, logger):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = load_config(args.config)
    simulation_config = copy.deepcopy(config["simulation"])

    with open(args.input_path, "rb") as f:
        episode_data = pkl.load(f)

    annotated = {}
    for key, episode_cache in episode_data.items():
        logger.info("Annotating episode %s...", key)
        episode_copy = copy.deepcopy(episode_cache)
        label_info = dict(episode_copy.get("label_info", {}))
        shared_info = run_expert_shared_episode(simulation_config, episode_copy)
        label_info.update(shared_info)
        episode_copy["label_info"] = label_info
        annotated[key] = episode_copy
        logger.info("Episode %s shared oracle label: %s", key, shared_info)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "wb") as f:
        pkl.dump(annotated, f)
    logger.info("Saved annotated dataset to %s", args.output_path)


if __name__ == "__main__":
    args = parse_arguments()
    logger = setup_logger(log_dir=args.log_dir, log_name=args.exp)
    main(args, logger)
