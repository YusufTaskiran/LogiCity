import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


RUN_SPECS = {
    "ppo": {
        "exp_prefix": "easy_multi_ppo",
        "checkpoint": ROOT / "checkpoints" / "experiment_family_1" / "easy_lite_ppo_seed101" / "best_model.zip",
        "config_by_k": {
            1: ROOT / "config" / "tasks" / "Nav" / "easy" / "algo" / "multi_shared_ppo_test_k1.yaml",
            2: ROOT / "config" / "tasks" / "Nav" / "easy" / "algo" / "multi_shared_ppo_test_k2.yaml",
            3: ROOT / "config" / "tasks" / "Nav" / "easy" / "algo" / "multi_shared_ppo_test_k3.yaml",
        },
        "result_dir_fmt": "ppo_k{k}",
    },
    "plpg_coarse": {
        "exp_prefix": "easy_multi_plpg",
        "checkpoint": ROOT / "checkpoints" / "experiment_family_1" / "easy_lite_plpg_seed101" / "best_model.zip",
        "config_by_k": {
            1: ROOT / "config" / "tasks" / "Nav" / "easy" / "algo" / "multi_shared_plpg_test_k1.yaml",
            2: ROOT / "config" / "tasks" / "Nav" / "easy" / "algo" / "multi_shared_plpg_test_k2.yaml",
            3: ROOT / "config" / "tasks" / "Nav" / "easy" / "algo" / "multi_shared_plpg_test_k3.yaml",
        },
        "result_dir_fmt": "plpg_k{k}",
    },
    "plpg_fine": {
        "exp_prefix": "easy_multi_plpg_fine",
        "checkpoint": ROOT / "checkpoints" / "experiment_family_1" / "easy_lite_plpg_seed101" / "best_model.zip",
        "config_by_k": {
            1: ROOT / "config" / "tasks" / "Nav" / "easy" / "algo" / "multi_shared_plpg_fine_test_k1.yaml",
            2: ROOT / "config" / "tasks" / "Nav" / "easy" / "algo" / "multi_shared_plpg_fine_test_k2.yaml",
            3: ROOT / "config" / "tasks" / "Nav" / "easy" / "algo" / "multi_shared_plpg_fine_test_k3.yaml",
        },
        "result_dir_fmt": "plpg_fine_k{k}",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Family 1 PPO/PLPG evaluations for K=1,2,3.")
    parser.add_argument(
        "--methods",
        default="ppo,plpg_coarse,plpg_fine",
        help="Comma-separated subset of: ppo,plpg_coarse,plpg_fine",
    )
    parser.add_argument(
        "--ks",
        default="1,2,3",
        help="Comma-separated subset of K values, e.g. 2 or 1,3",
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--results-root",
        default=str(ROOT / "results" / "experiment_family_1"),
        help="Directory under which per-run result folders are created.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for invoking main.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def parse_csv_set(raw_value, valid_values, cast=str):
    values = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        value = cast(item)
        if value not in valid_values:
            raise ValueError(f"Unsupported value: {value}. Expected one of {sorted(valid_values)}")
        values.append(value)
    return values


def build_command(method_name, k_value, seed, results_root, python_exe):
    spec = RUN_SPECS[method_name]
    config_path = spec["config_by_k"][k_value]
    checkpoint_path = spec["checkpoint"]
    result_dir = Path(results_root) / spec["result_dir_fmt"].format(k=k_value)
    exp_name = f"{spec['exp_prefix']}_k{k_value}_seed{seed}"
    return [
        python_exe,
        "main.py",
        "--use_gym",
        "--config",
        str(config_path.relative_to(ROOT)),
        "--exp",
        exp_name,
        "--seed",
        str(seed),
        "--checkpoint_path",
        str(checkpoint_path.relative_to(ROOT)),
        "--log_dir",
        str(result_dir.relative_to(ROOT)),
    ]


def main():
    args = parse_args()
    methods = parse_csv_set(args.methods, set(RUN_SPECS.keys()))
    ks = parse_csv_set(args.ks, {1, 2, 3}, cast=int)

    for method_name in methods:
        for k_value in ks:
            command = build_command(method_name, k_value, args.seed, args.results_root, args.python)
            print(" ".join(command))
            if args.dry_run:
                continue
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
