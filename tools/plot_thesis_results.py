import argparse
import csv
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt


@dataclass
class Series:
    label: str
    points: list[tuple[float, float]]
    color: str


PALETTE = {
    "ppo": "#1f77b4",
    "dls": "#d62728",
    "pls": "#2ca02c",
    "expert": "#9467bd",
}


def infer_method_label(path: str) -> str:
    name = os.path.basename(path).lower()
    if "pls" in name:
        return "PPO + PLS"
    if "dls" in name:
        return "PPO + DLS"
    if "expert" in name:
        return "Expert"
    return "PPO"


def infer_color(label: str) -> str:
    lowered = label.lower()
    if "pls" in lowered:
        return PALETTE["pls"]
    if "dls" in lowered:
        return PALETTE["dls"]
    if "expert" in lowered:
        return PALETTE["expert"]
    return PALETTE["ppo"]


def read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_named_paths(items: list[str] | None) -> list[tuple[str | None, str]]:
    if not items:
        return []
    result = []
    for item in items:
        if "=" in item:
            label, path = item.split("=", 1)
            result.append((label, path))
        else:
            result.append((None, item))
    return result


def autodiscover_training() -> list[tuple[str, str]]:
    patterns = [
        ("PPO", "checkpoints/thesis_uncertain_ppo_single/thesis_uncertain_ppo_single_metrics.csv"),
        ("PPO + DLS", "checkpoints/thesis_uncertain_ppo_dls_single/thesis_uncertain_ppo_dls_single_metrics.csv"),
        ("PPO + PLS", "checkpoints/thesis_uncertain_ppo_pls_single/thesis_uncertain_ppo_pls_single_metrics.csv"),
    ]
    return [(label, path) for label, path in patterns if os.path.exists(path)]


def load_training_series(path: str, metric: str, label: str | None = None) -> Series:
    rows = read_csv_rows(path)
    points: list[tuple[float, float]] = []
    for row in rows:
        step = to_float(row.get("step"))
        value = to_float(row.get(metric))
        if step is None or value is None:
            continue
        points.append((step, value))
    series_label = label or infer_method_label(path)
    return Series(series_label, points, infer_color(series_label))


def load_latest_training_row(path: str, label: str | None = None) -> dict[str, float | str | None]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    latest = rows[-1]
    series_label = label or infer_method_label(path)
    return {
        "label": series_label,
        "elapsed_wall_clock_seconds": to_float(latest.get("elapsed_wall_clock_seconds")),
        "seconds_per_1k_timesteps": to_float(latest.get("seconds_per_1k_timesteps")),
        "train_fail": to_float(latest.get("train_fail")),
        "train_timeout": to_float(latest.get("train_timeout")),
        "train_success": to_float(latest.get("train_success")),
    }


def load_test_summary(path: str, label: str | None = None) -> dict[str, float | str | None]:
    rows = read_csv_rows(path)
    summary = None
    for row in rows:
        if row.get("row_type") == "summary":
            summary = row
            break
    if summary is None and rows:
        summary = rows[-1]
    if summary is None:
        raise ValueError(f"No rows found in {path}")
    series_label = label or infer_method_label(path)
    return {
        "label": series_label,
        "tsr": to_float(summary.get("tsr")),
        "dsr": to_float(summary.get("dsr")),
        "fail": to_float(summary.get("fail") or summary.get("fail_episodes")),
        "timeout": to_float(summary.get("timeout") or summary.get("timeout_episodes")),
        "mean_reward": to_float(summary.get("mean_reward")),
        "eval_wall_clock_seconds": to_float(summary.get("eval_wall_clock_seconds")),
    }


def save_line_plot(series_list: list[Series], title: str, ylabel: str, output_path: str) -> None:
    series_list = [series for series in series_list if series.points]
    if not series_list:
        return
    plt.figure(figsize=(10, 5))
    for series in series_list:
        xs = [x for x, _ in series.points]
        ys = [y for _, y in series.points]
        plt.plot(xs, ys, marker="o", linewidth=2, markersize=4, label=series.label, color=series.color)
    plt.title(title)
    plt.xlabel("Training timesteps")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_grouped_bar_chart(rows: list[dict[str, float | str | None]], metrics: list[str], title: str, output_path: str) -> None:
    if not rows:
        return
    labels = [str(row["label"]) for row in rows]
    colors = [infer_color(label) for label in labels]
    x = list(range(len(metrics)))
    width = 0.8 / max(1, len(rows))

    plt.figure(figsize=(11, 5))
    for idx, row in enumerate(rows):
        values = []
        for metric in metrics:
            value = row.get(metric)
            values.append(0.0 if value is None else float(value))
        offsets = [pos - 0.4 + width / 2 + idx * width for pos in x]
        plt.bar(offsets, values, width=width, label=labels[idx], color=colors[idx])

    plt.xticks(x, metrics)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_summary_csv(path: str, rows: list[dict[str, float | str | None]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis result plots as PNG files using matplotlib.")
    parser.add_argument(
        "--training",
        action="append",
        help="Training CSV input as LABEL=path or just path. Can be passed multiple times. If omitted, uncertain-sensor defaults are auto-discovered.",
    )
    parser.add_argument(
        "--single-test",
        action="append",
        help="Single-agent final test CSV input as LABEL=path or just path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default="vis/thesis_results",
        help="Directory where plots and summaries are written.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    training_inputs = parse_named_paths(args.training)
    if not training_inputs:
        training_inputs = autodiscover_training()

    tsr_series = [load_training_series(path, "tsr", label) for label, path in training_inputs]
    dsr_series = [load_training_series(path, "dsr", label) for label, path in training_inputs]
    reward_series = [load_training_series(path, "mean_reward", label) for label, path in training_inputs]
    train_fail_series = [load_training_series(path, "train_fail", label) for label, path in training_inputs]
    train_fail_delta_series = [load_training_series(path, "train_fail_since_last_eval", label) for label, path in training_inputs]
    runtime_rows = [load_latest_training_row(path, label) for label, path in training_inputs]

    save_line_plot(tsr_series, "TSR Learning Curve", "TSR", os.path.join(args.output_dir, "tsr_learning_curve.png"))
    save_line_plot(dsr_series, "DSR Learning Curve", "DSR", os.path.join(args.output_dir, "dsr_learning_curve.png"))
    save_line_plot(reward_series, "Mean Reward Learning Curve", "Mean reward", os.path.join(args.output_dir, "mean_reward_learning_curve.png"))
    save_line_plot(train_fail_series, "Cumulative Training Failures", "train_fail", os.path.join(args.output_dir, "train_fail_cumulative.png"))
    save_line_plot(train_fail_delta_series, "Training Failures Since Last Eval", "train_fail_since_last_eval", os.path.join(args.output_dir, "train_fail_since_last_eval.png"))
    save_grouped_bar_chart(
        runtime_rows,
        ["elapsed_wall_clock_seconds", "seconds_per_1k_timesteps"],
        "Training Runtime Overhead",
        os.path.join(args.output_dir, "runtime_overhead.png"),
    )
    save_summary_csv(os.path.join(args.output_dir, "training_runtime_summary.csv"), runtime_rows)

    single_test_inputs = parse_named_paths(args.single_test)
    if single_test_inputs:
        single_test_rows = [load_test_summary(path, label) for label, path in single_test_inputs]
        save_grouped_bar_chart(
            single_test_rows,
            ["tsr", "dsr", "fail", "timeout", "mean_reward"],
            "Final Single-Agent Test Summary",
            os.path.join(args.output_dir, "single_agent_test_summary.png"),
        )
        save_summary_csv(os.path.join(args.output_dir, "single_agent_test_summary.csv"), single_test_rows)

    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
