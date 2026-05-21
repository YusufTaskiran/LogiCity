import argparse
import copy
import os
import yaml
import json
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv

from logicity.rl_agent.alg.centralized_joint_pls_ppo import CentralizedJointPLSPPO
from logicity.rl_agent.policy.neural import MLPFeatureExtractor
from logicity.utils.centralized_joint_eval import (
    evaluate_centralized_joint_model,
    load_episode_data,
    write_episode_csv,
    write_summary_csv,
)
from logicity.utils.joint_two_car_vec_env import CentralizedJointTwoCarEnv
from logicity.utils.logger import setup_logger


def _jsonify_debug_value(value):
    if isinstance(value, dict):
        return {str(k): _jsonify_debug_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_debug_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonify_debug_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate centralized two-car Joint PLS.")
    parser.add_argument("--config", default="config/tasks/Nav/thesis/algo/centralized_joint_pls_two_cars_train.yaml")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--episode_data", required=True)
    parser.add_argument("--exp", default="thesis_centralized_joint_pls_eval")
    parser.add_argument("--debug_shield", action="store_true")
    parser.add_argument("--debug_episode", default=None)
    parser.add_argument("--debug_steps", type=int, default=3)
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def main():
    args = parse_args()
    logger = setup_logger(log_dir="./log_rl", log_name=args.exp)
    config = load_config(args.config)
    simulation_config = copy.deepcopy(config["simulation"])
    shield_cfg = copy.deepcopy(config.get("shield", {}))
    hyperparameters = copy.deepcopy(config["stable_baselines"]["hyperparameters"])
    logger.info("Starting centralized joint PLS evaluation: exp=%s", args.exp)
    logger.info("Checkpoint: %s", args.checkpoint_path)
    logger.info("Episode data: %s", args.episode_data)

    env = DummyVecEnv(
        [
            lambda: CentralizedJointTwoCarEnv(
                simulation_config=simulation_config,
                shield_config=shield_cfg,
                seed=1,
            )
        ]
    )
    model = CentralizedJointPLSPPO.load(
        args.checkpoint_path,
        env=env,
        **hyperparameters,
        policy_kwargs={
            "features_extractor_class": MLPFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 32},
        },
    )

    episode_data = load_episode_data(args.episode_data)
    if args.debug_shield:
        debug_episode = args.debug_episode
        if debug_episode is None:
            debug_episode = next(iter(episode_data.keys()))
        elif debug_episode not in episode_data:
            try:
                numeric_debug_episode = int(debug_episode)
            except (TypeError, ValueError):
                numeric_debug_episode = None
            if numeric_debug_episode in episode_data:
                debug_episode = numeric_debug_episode
        env_debug = CentralizedJointTwoCarEnv(
            simulation_config=simulation_config,
            shield_config=shield_cfg,
            fixed_episode_cache=episode_data[debug_episode],
            seed=1,
        )
        obs, _ = env_debug.reset()
        debug_rows = []
        done = False
        step = 0
        while (not done) and (step < args.debug_steps):
            snapshot = model.debug_action_snapshot(obs, env_source=env_debug)
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env_debug.step(int(action))
            debug_rows.append(
                {
                    "episode": debug_episode,
                    "step": step,
                    "chosen_joint_action": int(action),
                    "reward": float(reward),
                    "done": bool(terminated or truncated),
                    "info": info,
                    "snapshot": snapshot,
                }
            )
            obs = next_obs
            done = bool(terminated or truncated)
            step += 1
        env_debug.close()
        logger.info("Collected shield debug snapshots for episode %s", debug_episode)

    episode_rows, summary_row = evaluate_centralized_joint_model(
        model=model,
        simulation_config=simulation_config,
        shield_config=shield_cfg,
        episode_data=episode_data,
    )

    results_dir = os.path.join("results", args.exp)
    os.makedirs(results_dir, exist_ok=True)
    write_episode_csv(os.path.join(results_dir, "episode_metrics.csv"), episode_rows)
    write_summary_csv(os.path.join(results_dir, "summary_metrics.csv"), summary_row)
    if args.debug_shield:
        with open(os.path.join(results_dir, "shield_debug.json"), "w", encoding="utf-8") as f:
            json.dump(_jsonify_debug_value(debug_rows), f, indent=2)
        logger.info("Wrote shield debug JSON to %s", os.path.join(results_dir, "shield_debug.json"))
    logger.info("Wrote episode metrics to %s", os.path.join(results_dir, "episode_metrics.csv"))
    logger.info("Wrote summary metrics to %s", os.path.join(results_dir, "summary_metrics.csv"))
    env.close()


if __name__ == "__main__":
    main()
