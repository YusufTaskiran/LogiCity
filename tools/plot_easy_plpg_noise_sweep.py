import argparse
import csv
import os
from collections import OrderedDict

import matplotlib.pyplot as plt


METRICS = [
    ("tsr", "TSR"),
    ("mean_reward", "Mean Reward"),
    ("failure_rate", "Failure Rate"),
    ("timeout_rate", "Timeout Rate"),
    ("rollout_failures_since_last_eval", "Rollout Failures Since Last Eval"),
    ("rollout_timeouts_since_last_eval", "Rollout Timeouts Since Last Eval"),
    ("mean_base_policy_safe_prob", "Mean Base Policy Safety"),
    ("mean_shielded_policy_safe_prob", "Mean Shielded Policy Safety"),
    ("shield_intervention_rate", "Shield Intervention Rate"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot easy PLPG soft-noise sweep results.")
    parser.add_argument("--csvs", nargs="+", required=True, help="Eval CSVs for different noise levels.")
    parser.add_argument("--labels", nargs="+", required=True, help="Display labels matching --csvs order.")
    parser.add_argument("--output_dir", required=True, help="Directory to write plots.")
    return parser.parse_args()


def to_float(value, default=0.0):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def read_rows(path):
    with open(path, "r", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def make_plot(series_by_label, metric_key, metric_label, output_path):
    plt.figure(figsize=(8, 5))
    for label, rows in series_by_label.items():
        xs = [to_float(row["timestep"]) for row in rows]
        ys = [to_float(row.get(metric_key), 0.0) for row in rows]
        plt.plot(xs, ys, marker="o", label=label)
    plt.xlabel("Training Timesteps")
    plt.ylabel(metric_label)
    plt.title(f"Easy PLPG Noise Sweep: {metric_label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    args = parse_args()
    if len(args.csvs) != len(args.labels):
        raise ValueError("--csvs and --labels must have the same length.")
    ensure_dir(args.output_dir)

    series_by_label = OrderedDict()
    for label, path in zip(args.labels, args.csvs):
        series_by_label[label] = read_rows(path)

    for metric_key, metric_label in METRICS:
        output_path = os.path.join(args.output_dir, f"{metric_key}.png")
        make_plot(series_by_label, metric_key, metric_label, output_path)


if __name__ == "__main__":
    main()
