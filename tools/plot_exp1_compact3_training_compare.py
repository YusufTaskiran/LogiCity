import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
K_COLORS = {
    "K1": "#4c78a8",
    "K2": "#f58518",
    "K3": "#54a24b",
}
ACTION_COLORS = {
    "slow": "#9c755f",
    "normal": "#72b7b2",
    "fast": "#4c78a8",
    "stop": "#e45756",
}
METRICS = [
    ("tsr", "TSR", (0.0, 1.05)),
    ("timeout_rate", "Timeout Rate", (0.0, 1.05)),
    ("failure_rate", "Failure Rate", (0.0, 1.05)),
    ("traffic_violation_count", "Traffic Violation Count", None),
    ("rollout_traffic_violations_since_last_eval", "Traffic Violations Since Last Eval", None),
    ("rollout_failures_since_last_eval", "Rollout Failures Since Last Eval", None),
]
BEST_FIELDS = [
    "k",
    "best_eval_index",
    "best_timestep",
    "best_tsr",
    "best_mean_reward",
    "best_timeout_rate",
    "best_failure_rate",
    "best_traffic_violation_count",
    "best_rollout_traffic_violations_since_last_eval",
    "best_rollout_failures_since_last_eval",
    "action_slow_count",
    "action_normal_count",
    "action_fast_count",
    "action_stop_count",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare exp1 compact3 K1/K2/K3 training curves and best-model actions."
    )
    parser.add_argument(
        "--k1-csv",
        default=str(ROOT / "evaluation_results" / "metrics" / "exp1_compact3_easy_plpg_k1_v7" / "training.csv"),
    )
    parser.add_argument(
        "--k2-csv",
        default=str(ROOT / "evaluation_results" / "metrics" / "exp1_compact3_easy_plpg_k2_v7" / "training.csv"),
    )
    parser.add_argument(
        "--k3-csv",
        default=str(ROOT / "evaluation_results" / "metrics" / "exp1_compact3_easy_plpg_k3_v7" / "training.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "plots" / "exp1_compact3_training_compare"),
    )
    return parser.parse_args()


def to_float(value, default=0.0):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_rows(path):
    with open(path, "r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def best_row(rows):
    return max(
        rows,
        key=lambda row: (
            to_float(row.get("tsr")),
            to_float(row.get("mean_reward")),
            -to_float(row.get("timeout_rate")),
            -to_float(row.get("failure_rate")),
            to_float(row.get("timestep")),
        ),
    )


def ensure_dir(path_str):
    Path(path_str).mkdir(parents=True, exist_ok=True)


def plot_metric_grid(series_map, output_dir):
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for ax, (metric, title, ylim) in zip(axes.flat, METRICS):
        for label, rows in series_map.items():
            xs = [to_float(row.get("timestep")) for row in rows]
            ys = [to_float(row.get(metric)) for row in rows]
            ax.plot(xs, ys, marker="o", linewidth=2.0, color=K_COLORS[label], label=label)
        ax.set_title(title)
        ax.set_xlabel("Eval Steps")
        ax.grid(alpha=0.25)
        if ylim is not None:
            ax.set_ylim(*ylim)
    axes[0, 0].legend()
    fig.suptitle("Experiment 1 Compact3 Training Comparison", y=0.995)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "training_metrics_grid.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metric_panels(series_map, output_dir):
    for metric, title, ylim in METRICS:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for label, rows in series_map.items():
            xs = [to_float(row.get("timestep")) for row in rows]
            ys = [to_float(row.get(metric)) for row in rows]
            ax.plot(xs, ys, marker="o", linewidth=2.2, color=K_COLORS[label], label=label)
        ax.set_title(title)
        ax.set_xlabel("Eval Steps")
        ax.grid(alpha=0.25)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(output_dir) / f"{metric}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_best_action_hist(best_rows, output_dir):
    ks = list(best_rows.keys())
    actions = ["slow", "normal", "fast", "stop"]
    x = range(len(ks))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for idx, action in enumerate(actions):
        xs = [pos + (idx - 1.5) * width for pos in x]
        ys = [to_float(best_rows[k].get(f"action_{action}_count")) for k in ks]
        ax.bar(xs, ys, width=width, color=ACTION_COLORS[action], label=action)
    ax.set_xticks(list(x))
    ax.set_xticklabels(ks)
    ax.set_xlabel("Best-Performing Model per K")
    ax.set_ylabel("Action Count During Eval")
    ax.set_title("Best-Model Action Histogram")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "best_model_action_histogram.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_eval_table(series_map, output_dir):
    output_path = Path(output_dir) / "training_eval_summary.csv"
    fields = ["k", "eval_index", "timestep"] + [metric for metric, _title, _ylim in METRICS]
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, rows in series_map.items():
            for row in rows:
                writer.writerow(
                    {
                        "k": label,
                        "eval_index": int(to_float(row.get("eval_index"))),
                        "timestep": int(to_float(row.get("timestep"))),
                        **{
                            metric: row.get(metric, "")
                            for metric, _title, _ylim in METRICS
                        },
                    }
                )


def write_best_table(best_rows, output_dir):
    output_path = Path(output_dir) / "best_model_summary.csv"
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BEST_FIELDS)
        writer.writeheader()
        for label, row in best_rows.items():
            writer.writerow(
                {
                    "k": label,
                    "best_eval_index": int(to_float(row.get("eval_index"))),
                    "best_timestep": int(to_float(row.get("timestep"))),
                    "best_tsr": row.get("tsr", ""),
                    "best_mean_reward": row.get("mean_reward", ""),
                    "best_timeout_rate": row.get("timeout_rate", ""),
                    "best_failure_rate": row.get("failure_rate", ""),
                    "best_traffic_violation_count": row.get("traffic_violation_count", ""),
                    "best_rollout_traffic_violations_since_last_eval": row.get(
                        "rollout_traffic_violations_since_last_eval", ""
                    ),
                    "best_rollout_failures_since_last_eval": row.get(
                        "rollout_failures_since_last_eval", ""
                    ),
                    "action_slow_count": row.get("action_slow_count", ""),
                    "action_normal_count": row.get("action_normal_count", ""),
                    "action_fast_count": row.get("action_fast_count", ""),
                    "action_stop_count": row.get("action_stop_count", ""),
                }
            )


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    series_map = {
        "K1": read_rows(args.k1_csv),
        "K2": read_rows(args.k2_csv),
        "K3": read_rows(args.k3_csv),
    }
    best_rows = {label: best_row(rows) for label, rows in series_map.items()}

    plot_metric_grid(series_map, args.output_dir)
    plot_metric_panels(series_map, args.output_dir)
    plot_best_action_hist(best_rows, args.output_dir)
    write_eval_table(series_map, args.output_dir)
    write_best_table(best_rows, args.output_dir)

    print(f"Wrote comparison artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
