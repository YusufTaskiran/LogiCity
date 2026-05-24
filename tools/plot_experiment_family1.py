import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt


CORE_CURVE_METRICS = [
    ("tsr", "TSR"),
    ("failure_rate", "Failure Rate"),
    ("timeout_rate", "Timeout Rate"),
    ("mean_reward", "Mean Reward"),
]

SAFETY_BEHAVIOR_METRICS = [
    ("action_stop_count", "Stop Count"),
    ("decision_succ_action_stop", "Stop Decision Success"),
]

ROLLOUT_METRICS = [
    ("rollout_failures_since_last_eval", "Rollout Failures Since Last Eval"),
    ("rollout_timeouts_since_last_eval", "Rollout Timeouts Since Last Eval"),
    ("rollout_successes_since_last_eval", "Rollout Successes Since Last Eval"),
]

PLPG_METRICS = [
    ("mean_base_policy_safe_prob", "Base Policy Safety"),
    ("mean_shielded_policy_safe_prob", "Shielded Policy Safety"),
    ("mean_safety_gain", "Safety Gain"),
    ("shield_intervention_rate", "Shield Intervention Rate"),
    ("mean_policy_kl_base_to_shielded", "KL(Base || Shielded)"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Experiment Family 1 results from eval/test CSVs.")
    parser.add_argument("--eval_csvs", nargs="+", required=True, help="Validation/training eval CSV paths.")
    parser.add_argument("--test_csvs", nargs="*", default=[], help="Held-out test CSV paths.")
    parser.add_argument("--output_dir", required=True, help="Directory to save plots.")
    return parser.parse_args()


def read_csv_rows(paths):
    rows = []
    for path in paths:
        with open(path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["_source_path"] = path
                rows.append(row)
    return rows


def to_float(value, default=0.0):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def canonical_method(method_name):
    name = str(method_name or "").lower()
    if "plpg" in name:
        return "PLPG-PPO"
    if "ppo" in name:
        return "PPO"
    return str(method_name or "unknown")


def canonical_difficulty(value):
    value = str(value or "").lower()
    for difficulty in ["easy", "medium", "hard"]:
        if difficulty in value:
            return difficulty
    return value or "unknown"


def group_eval_rows(rows):
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        difficulty = canonical_difficulty(row.get("difficulty"))
        method = canonical_method(row.get("method"))
        timestep = int(to_float(row.get("timestep"), 0))
        grouped[difficulty][method][timestep].append(row)
    return grouped


def group_eval_rows_by_wall_clock(rows):
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        difficulty = canonical_difficulty(row.get("difficulty"))
        method = canonical_method(row.get("method"))
        wall_clock_time_sec = to_float(row.get("wall_clock_time_sec"), 0.0)
        grouped[difficulty][method][wall_clock_time_sec].append(row)
    return grouped


def group_test_rows(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        difficulty = canonical_difficulty(row.get("difficulty"))
        method = canonical_method(row.get("method"))
        grouped[difficulty][method].append(row)
    return grouped


def mean_std(values):
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def make_curve_plot(grouped, metric_key, metric_label, output_path):
    make_curve_plot_with_x(grouped, metric_key, metric_label, output_path, "Training Timesteps")


def make_curve_plot_with_x(grouped, metric_key, metric_label, output_path, x_label):
    plt.figure(figsize=(8, 5))
    for method, timestep_rows in sorted(grouped.items()):
        xs = sorted(timestep_rows.keys())
        ys = []
        y_low = []
        y_high = []
        for x in xs:
            vals = [to_float(row.get(metric_key), 0.0) for row in timestep_rows[x]]
            mean, std = mean_std(vals)
            ys.append(mean)
            y_low.append(mean - std)
            y_high.append(mean + std)
        plt.plot(xs, ys, label=method)
        plt.fill_between(xs, y_low, y_high, alpha=0.2)
    plt.xlabel(x_label)
    plt.ylabel(metric_label)
    plt.title(metric_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def make_plpg_safety_plot(grouped, output_path):
    plt.figure(figsize=(8, 5))
    for metric_key, metric_label in [
        ("mean_base_policy_safe_prob", "Base Safety"),
        ("mean_shielded_policy_safe_prob", "Shielded Safety"),
        ("mean_safety_gain", "Safety Gain"),
    ]:
        xs = sorted(grouped.keys())
        ys = []
        for x in xs:
            vals = [to_float(row.get(metric_key), 0.0) for row in grouped[x]]
            mean, _ = mean_std(vals)
            ys.append(mean)
        plt.plot(xs, ys, label=metric_label)
    plt.xlabel("Training Timesteps")
    plt.ylabel("Probability / Gain")
    plt.title("PLPG Safety Diagnostics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def make_test_bar_chart(grouped, metric_keys, output_path, title):
    methods = sorted(grouped.keys())
    labels = [label for _, label in metric_keys]
    x_positions = list(range(len(labels)))
    width = 0.35 if len(methods) <= 2 else 0.8 / max(len(methods), 1)

    plt.figure(figsize=(9, 5))
    for idx, method in enumerate(methods):
        means = []
        for metric_key, _metric_label in metric_keys:
            vals = [to_float(row.get(metric_key), 0.0) for row in grouped[method]]
            mean, _ = mean_std(vals)
            means.append(mean)
        offsets = [x + (idx - (len(methods) - 1) / 2.0) * width for x in x_positions]
        plt.bar(offsets, means, width=width, label=method)
    plt.xticks(x_positions, labels)
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    eval_rows = read_csv_rows(args.eval_csvs)
    eval_grouped = group_eval_rows(eval_rows)
    eval_grouped_wall_clock = group_eval_rows_by_wall_clock(eval_rows)

    for difficulty, difficulty_group in eval_grouped.items():
        for metric_key, metric_label in CORE_CURVE_METRICS:
            make_curve_plot(
                difficulty_group,
                metric_key,
                metric_label,
                os.path.join(args.output_dir, f"{difficulty}_{metric_key}_curve.png"),
            )
            make_curve_plot_with_x(
                eval_grouped_wall_clock[difficulty],
                metric_key,
                f"{metric_label} (Wall Clock)",
                os.path.join(args.output_dir, f"{difficulty}_{metric_key}_curve_wall_clock.png"),
                "Wall-Clock Time (s)",
            )
        for metric_key, metric_label in SAFETY_BEHAVIOR_METRICS:
            make_curve_plot(
                difficulty_group,
                metric_key,
                metric_label,
                os.path.join(args.output_dir, f"{difficulty}_{metric_key}_curve.png"),
            )
        for metric_key, metric_label in ROLLOUT_METRICS:
            make_curve_plot(
                difficulty_group,
                metric_key,
                metric_label,
                os.path.join(args.output_dir, f"{difficulty}_{metric_key}_curve.png"),
            )
        if "PLPG-PPO" in difficulty_group:
            make_plpg_safety_plot(
                difficulty_group["PLPG-PPO"],
                os.path.join(args.output_dir, f"{difficulty}_plpg_safety_diagnostics.png"),
            )
            for metric_key, metric_label in PLPG_METRICS[3:]:
                make_curve_plot(
                    {"PLPG-PPO": difficulty_group["PLPG-PPO"]},
                    metric_key,
                    metric_label,
                    os.path.join(args.output_dir, f"{difficulty}_{metric_key}_curve.png"),
                )

    if args.test_csvs:
        test_rows = read_csv_rows(args.test_csvs)
        test_grouped = group_test_rows(test_rows)
        for difficulty, difficulty_group in test_grouped.items():
            make_test_bar_chart(
                difficulty_group,
                [
                    ("tsr", "TSR"),
                    ("failure_rate", "Failure Rate"),
                    ("timeout_rate", "Timeout Rate"),
                    ("mean_reward", "Mean Reward"),
                ],
                os.path.join(args.output_dir, f"{difficulty}_final_test_summary.png"),
                f"{difficulty.capitalize()} Final Test Summary",
            )


if __name__ == "__main__":
    main()
