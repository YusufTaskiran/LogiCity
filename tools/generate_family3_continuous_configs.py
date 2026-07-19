import copy
import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "config" / "tasks" / "Nav"
KS = [5, 10, 15, 20]
DEFAULT_CONTINUOUS_EVAL = {
    "enabled": True,
    "stop_mode": "goal_quota_per_rl_agent",
    "total_steps": None,
    "resolved_goal_quota_per_agent": 20,
    "reset_on_failure": True,
    "visualize": {
        "enabled": False,
        "frame_every": 10,
        "resolution": 1000,
        "agent_layers": -1,
    },
}
FULL_MAP_REGION = 250

METHODS = {
    "ppo": {
        "template": "multi_shared_5x5_ppo_test_k15.yaml",
        "output": "multi_shared_5x5_ppo_test_k{K}.yaml",
    },
    "plpg": {
        "template": "multi_shared_5x5_plpg_test_k15.yaml",
        "output": "multi_shared_5x5_plpg_test_k{K}.yaml",
    },
    "plpg_fine": {
        "template": "multi_shared_5x5_plpg_fine_test_k15.yaml",
        "output": "multi_shared_5x5_plpg_fine_test_k{K}.yaml",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate family-3 continuous evaluation configs.")
    parser.add_argument(
        "--stop_mode",
        choices=["goal_quota_per_rl_agent", "time_budget"],
        default=DEFAULT_CONTINUOUS_EVAL["stop_mode"],
        help="Continuous evaluation stopping rule.",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=None,
        help="Total rollout steps for time_budget mode.",
    )
    parser.add_argument(
        "--resolved_goal_quota_per_agent",
        type=int,
        default=DEFAULT_CONTINUOUS_EVAL["resolved_goal_quota_per_agent"],
        help="Resolved goal quota per RL agent for quota-based mode.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable rollout frame capture.",
    )
    parser.add_argument(
        "--frame_every",
        type=int,
        default=DEFAULT_CONTINUOUS_EVAL["visualize"]["frame_every"],
        help="Capture every Nth frame when visualization is enabled.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_CONTINUOUS_EVAL["visualize"]["resolution"],
        help="Visualization resolution.",
    )
    parser.add_argument(
        "--agent_layers",
        type=int,
        default=DEFAULT_CONTINUOUS_EVAL["visualize"]["agent_layers"],
        help="Agent layer selection for visualization.",
    )
    parser.add_argument(
        "--reset_on_failure",
        type=lambda value: value.lower() in {"1", "true", "yes", "y"},
        default=DEFAULT_CONTINUOUS_EVAL["reset_on_failure"],
        help="Whether to reset deployments after failure.",
    )
    return parser.parse_args()


def build_continuous_eval(args):
    cfg = copy.deepcopy(DEFAULT_CONTINUOUS_EVAL)
    cfg["stop_mode"] = args.stop_mode
    cfg["total_steps"] = args.total_steps if args.stop_mode == "time_budget" else None
    cfg["resolved_goal_quota_per_agent"] = (
        args.resolved_goal_quota_per_agent if args.stop_mode == "goal_quota_per_rl_agent" else None
    )
    cfg["reset_on_failure"] = args.reset_on_failure
    cfg["visualize"]["enabled"] = args.visualize
    cfg["visualize"]["frame_every"] = args.frame_every
    cfg["visualize"]["resolution"] = args.resolution
    cfg["visualize"]["agent_layers"] = args.agent_layers
    return cfg


def main():
    args = parse_args()
    continuous_eval = build_continuous_eval(args)
    for difficulty in ["easy", "medium", "hard"]:
        algo_dir = CONFIG_ROOT / difficulty / "algo"
        for method_meta in METHODS.values():
            template_path = algo_dir / method_meta["template"]
            with template_path.open("r", encoding="utf-8") as handle:
                template_cfg = yaml.safe_load(handle)

            for k in KS:
                cfg = copy.deepcopy(template_cfg)
                cfg["simulation"]["agent_region"] = copy.deepcopy(FULL_MAP_REGION)
                cfg["simulation"]["rl_agent"]["agent_names"] = [f"Car_{i}" for i in range(1, k + 1)]
                cfg["stable_baselines"]["train"] = False
                cfg["stable_baselines"].pop("episode_data", None)
                cfg["stable_baselines"]["continuous_eval"] = copy.deepcopy(continuous_eval)

                output_path = algo_dir / method_meta["output"].format(K=k)
                with output_path.open("w", encoding="utf-8") as handle:
                    yaml.safe_dump(cfg, handle, sort_keys=False)
                print(f"Wrote {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
