import argparse
import copy
import os
import yaml

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from logicity.rl_agent.alg.centralized_joint_pls_ppo import CentralizedJointPLSPPO
from logicity.utils.centralized_joint_callback import CentralizedJointEvalCheckpointCallback
from logicity.utils.joint_two_car_vec_env import CentralizedJointTwoCarEnv
from logicity.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train centralized two-car Joint PLS.")
    parser.add_argument("--config", default="config/tasks/Nav/thesis/algo/centralized_joint_pls_two_cars_train.yaml")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--log_dir", default="./log_rl")
    parser.add_argument("--exp", default="thesis_centralized_joint_pls_two_cars")
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--train_episode_data", default=None)
    parser.add_argument("--validation_episode_data", default=None)
    parser.add_argument("--debug_shield", action="store_true")
    parser.add_argument("--debug_steps", type=int, default=3)
    parser.add_argument("--debug_episode", default=None)
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def dynamic_import(module_name, class_name):
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def main():
    args = parse_args()
    logger = setup_logger(log_dir=args.log_dir, log_name=args.exp)
    config = load_config(args.config)
    simulation_config = copy.deepcopy(config["simulation"])
    shield_cfg = copy.deepcopy(config.get("shield", {}))
    rl_config = config["stable_baselines"]
    logger.info("Starting centralized joint PLS training: exp=%s seed=%s", args.exp, args.seed)
    logger.info("Config: %s", args.config)

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

    episode_data = (
        args.train_episode_data
        or rl_config.get("training_episode_data")
        or simulation_config["rl_agent"].get("training_episode_data")
    )
    validation_episode_data = args.validation_episode_data or rl_config.get("validation_episode_data")
    logger.info("Train episode data: %s", episode_data)
    logger.info("Validation episode data: %s", validation_episode_data)

    def make_env():
        return CentralizedJointTwoCarEnv(
            simulation_config=simulation_config,
            shield_config=shield_cfg,
            episode_data_path=episode_data if episode_data and os.path.isfile(episode_data) else None,
            seed=args.seed,
        )

    env = DummyVecEnv([make_env])

    save_cfg = config.get("eval_checkpoint", {})
    save_root = os.path.join(save_cfg.get("save_path", "./checkpoints"), args.exp)
    os.makedirs(save_root, exist_ok=True)
    checkpoint_save_path = os.path.join(save_root, "checkpoints", args.exp)
    if validation_episode_data and os.path.isfile(validation_episode_data):
        logger.info(
            "Validation callback enabled: eval_freq=%s save_freq=%s",
            int(save_cfg.get("eval_freq", save_cfg.get("save_freq", 500))),
            int(save_cfg.get("save_freq", 500)),
        )
        checkpoint_cb = CentralizedJointEvalCheckpointCallback(
            exp_name=args.exp,
            simulation_config=simulation_config,
            shield_config=shield_cfg,
            validation_episode_data=validation_episode_data,
            eval_freq=int(save_cfg.get("eval_freq", save_cfg.get("save_freq", 500))),
            result_path=save_root,
            debug_shield=bool(args.debug_shield),
            debug_steps=int(args.debug_steps),
            debug_episode=args.debug_episode,
            save_freq=int(save_cfg.get("save_freq", 500)),
            save_path=checkpoint_save_path,
            name_prefix=args.exp,
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
    else:
        logger.info("Validation callback disabled; only checkpoints will be saved.")
        checkpoint_cb = CheckpointCallback(
            save_freq=int(save_cfg.get("save_freq", 500)),
            save_path=checkpoint_save_path,
            name_prefix=args.exp,
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

    hyperparameters = copy.deepcopy(rl_config["hyperparameters"])
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        model = CentralizedJointPLSPPO.load(
            args.checkpoint_path,
            env=env,
            **hyperparameters,
            policy_kwargs=sb3_policy_kwargs,
        )
    else:
        model = CentralizedJointPLSPPO(
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
    logger.info("Finished centralized joint PLS training. Saved final model to %s", os.path.join(save_root, "best_model"))
    env.close()


if __name__ == "__main__":
    main()
