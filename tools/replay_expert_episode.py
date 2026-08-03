import argparse
import os
import pickle as pkl
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from logicity.utils.logger import setup_logger
from main import make_env
from tools.create_episode import (
    _done_to_bool,
    _log_predicate_trace,
    _log_step_trace,
    load_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Replay one cached episode using the expert policy.")
    parser.add_argument("--config", required=True, help="Config YAML used for the environment.")
    parser.add_argument("--episode_data", required=True, help="Path to the cached episode_data PKL.")
    parser.add_argument("--episode_id", type=int, required=True, help="Episode id inside the PKL.")
    parser.add_argument("--log_dir", default="./log_rl", help="Directory for logs and optional output PKL.")
    parser.add_argument("--exp", default="expert_replay", help="Experiment/log name prefix.")
    parser.add_argument("--trace_predicates", action="store_true", help="Print predicate trace every step.")
    parser.add_argument("--trace_steps", type=int, default=0, help="If >0, print step/action trace for the first N steps.")
    parser.add_argument("--save_world", action="store_true", help="Save the replayed world trace as <exp>_<episode_id>.pkl.")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(log_dir=args.log_dir, log_name=args.exp)

    config = load_config(args.config)
    simulation_config = config["simulation"]

    with open(args.episode_data, "rb") as f:
        episode_data = pkl.load(f)
    if args.episode_id not in episode_data:
        raise KeyError("Episode id {} not found in {}".format(args.episode_id, args.episode_data))

    episode_cache = episode_data[args.episode_id]
    logger.info("Loaded episode %s from %s", args.episode_id, args.episode_data)
    if "label_info" in episode_cache:
        logger.info("Episode label: %s", episode_cache["label_info"])

    eval_env, cached_observation = make_env(simulation_config, episode_cache, True)
    obs = eval_env.init()
    if args.save_world:
        cached_observation["Time_Obs"][0] = {
            "World": eval_env.env.city_grid.clone(),
            "Info": None,
        }

    max_steps = eval_env.horizon if hasattr(eval_env, "horizon") else int(
        (episode_cache.get("label_info") or {}).get("oracle_step", 75) * 2
    )

    done = False
    step = 0
    total_reward = 0.0

    while (not done) and (step < max_steps):
        if args.trace_steps > 0 and step < args.trace_steps:
            _log_step_trace(
                logger.info,
                step,
                getattr(eval_env, "last_planner_actions", {}) if hasattr(eval_env, "last_planner_actions") else {},
                getattr(eval_env, "last_agent_positions", {}) if hasattr(eval_env, "last_agent_positions") else {},
            )
            if args.trace_predicates:
                _log_predicate_trace(logger.info, step, eval_env)

        action = eval_env.expert_action
        step_result = eval_env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = _done_to_bool(terminated) or _done_to_bool(truncated)
        else:
            obs, reward, done, info = step_result
            done = _done_to_bool(done)

        step += 1
        total_reward += float(reward) if not hasattr(reward, "__len__") else float(sum(reward))

        if args.save_world:
            cached_observation["Time_Obs"][step] = {
                "World": eval_env.env.city_grid.clone(),
                "Info": info,
            }

    logger.info("Replay finished: steps=%s total_reward=%s done=%s", step, total_reward, done)
    logger.info("Final info: %s", info)

    if args.save_world:
        output_path = os.path.join(args.log_dir, "{}_{}.pkl".format(args.exp, args.episode_id))
        with open(output_path, "wb") as f:
            pkl.dump(cached_observation, f)
        logger.info("Saved replay world trace to %s", output_path)


if __name__ == "__main__":
    main()
