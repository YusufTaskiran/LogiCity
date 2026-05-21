import argparse
import copy
import os
import yaml

from stable_baselines3.common.callbacks import CheckpointCallback

from logicity.rl_agent.alg import JointPLSPPO
from logicity.utils.joint_two_car_vec_env import JointTwoCarVecEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train shared-policy two-car Joint PLS.")
    parser.add_argument("--config", default="config/tasks/Nav/thesis/algo/ppo_joint_pls_two_cars_train.yaml")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--log_dir", default="./log_rl")
    parser.add_argument("--exp", default="thesis_joint_pls_two_cars")
    parser.add_argument("--checkpoint_path", default=None)
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def dynamic_import(module_name, class_name):
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def main():
    args = parse_args()
    config = load_config(args.config)
    simulation_config = copy.deepcopy(config["simulation"])
    shield_cfg = copy.deepcopy(config.get("shield", {}))
    rl_config = config["stable_baselines"]

    policy_kwargs = rl_config["policy_kwargs"]
    if "features_extractor_module" in policy_kwargs:
        features_extractor_class = dynamic_import(
            policy_kwargs["features_extractor_module"],
            policy_kwargs["features_extractor_class"],
        )
        sb3_policy_kwargs = {
            "features_extractor_class": features_extractor_class,
            "features_extractor_kwargs": policy_kwargs["features_extractor_kwargs"],
        }
    else:
        sb3_policy_kwargs = policy_kwargs

    episode_data = rl_config.get("training_episode_data") or simulation_config["rl_agent"].get("training_episode_data")
    env = JointTwoCarVecEnv(
        simulation_config=simulation_config,
        shield_config=shield_cfg,
        episode_data_path=episode_data if episode_data and os.path.isfile(episode_data) else None,
        seed=args.seed,
    )

    save_cfg = config.get("eval_checkpoint", {})
    save_root = os.path.join(save_cfg.get("save_path", "./checkpoints"), args.exp)
    os.makedirs(save_root, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=int(save_cfg.get("save_freq", 500)),
        save_path=os.path.join(save_root, "checkpoints", args.exp),
        name_prefix=args.exp,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    hyperparameters = copy.deepcopy(rl_config["hyperparameters"])
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        model = JointPLSPPO.load(
            args.checkpoint_path,
            env=env,
            **hyperparameters,
            policy_kwargs=sb3_policy_kwargs,
        )
    else:
        model = JointPLSPPO(
            rl_config["policy_network"],
            env,
            **hyperparameters,
            policy_kwargs=sb3_policy_kwargs,
        )

    model.learn(
        total_timesteps=int(rl_config["total_timesteps"]),
        callback=checkpoint_cb,
        tb_log_name=args.exp,
    )
    model.save(os.path.join(save_root, "best_model"))
    env.close()


if __name__ == "__main__":
    main()

