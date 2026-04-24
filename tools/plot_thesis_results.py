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
    linestyle: str = "-"
    marker: str = "o"


PALETTE = {
    "ppo": "#1f77b4",
    "dls": "#d62728",
    "pls": "#2ca02c",
    "expert": "#9467bd",
}

VARIANT_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

SEED_COLORS = {
    "s1": "#1f77b4",
    "s2": "#ff7f0e",
    "s3": "#2ca02c",
}

SEED_LINESTYLES = {
    "s1": "-",
    "s2": "--",
    "s3": "-.",
}

SEED_MARKERS = {
    "s1": "o",
    "s2": "s",
    "s3": "^",
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
    for seed, color in SEED_COLORS.items():
        if seed in lowered:
            return color
    if "pls" in lowered:
        return PALETTE["pls"]
    if "dls" in lowered:
        return PALETTE["dls"]
    if "expert" in lowered:
        return PALETTE["expert"]
    return PALETTE["ppo"]


def infer_linestyle(label: str) -> str:
    lowered = label.lower()
    for seed, linestyle in SEED_LINESTYLES.items():
        if seed in lowered:
            return linestyle
    return "-"


def infer_marker(label: str) -> str:
    lowered = label.lower()
    for seed, marker in SEED_MARKERS.items():
        if seed in lowered:
            return marker
    return "o"


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


def parse_named_groups(items: list[str] | None) -> list[tuple[str, list[str]]]:
    if not items:
        return []
    result = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected LABEL=path1,path2,... format, got: {item}")
        label, paths_str = item.split("=", 1)
        paths = [path.strip() for path in paths_str.split(",") if path.strip()]
        if not paths:
            raise ValueError(f"No paths provided for grouped input: {item}")
        result.append((label.strip(), paths))
    return result


def autodiscover_training() -> list[tuple[str, str]]:
    patterns = [
        ("PPO", "checkpoints/thesis_ppo_single/thesis_ppo_single_metrics.csv"),
        ("PPO + DLS", "checkpoints/thesis_ppo_dls_single/thesis_ppo_dls_single_metrics.csv"),
        ("PPO + PLS", "checkpoints/thesis_ppo_pls_single/thesis_ppo_pls_single_metrics.csv"),
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
    return Series(
        series_label,
        points,
        infer_color(series_label),
        infer_linestyle(series_label),
        infer_marker(series_label),
    )


def load_cumulative_training_series(path: str, metric: str, label: str | None = None) -> Series:
    rows = read_csv_rows(path)
    points: list[tuple[float, float]] = []
    running_total = 0.0
    for row in rows:
        step = to_float(row.get("step"))
        value = to_float(row.get(metric))
        if step is None or value is None:
            continue
        running_total += value
        points.append((step, running_total))
    series_label = label or infer_method_label(path)
    return Series(
        series_label,
        points,
        infer_color(series_label),
        infer_linestyle(series_label),
        infer_marker(series_label),
    )


def average_series(paths: list[str], metric: str, label: str) -> Series:
    by_step: dict[float, list[float]] = {}
    for path in paths:
        rows = read_csv_rows(path)
        for row in rows:
            step = to_float(row.get("step"))
            value = to_float(row.get(metric))
            if step is None or value is None:
                continue
            by_step.setdefault(step, []).append(value)
    points = []
    for step in sorted(by_step.keys()):
        values = by_step[step]
        if values:
            points.append((step, sum(values) / len(values)))
    return Series(label, points, infer_color(label), "-", "o")


def average_cumulative_series(paths: list[str], metric: str, label: str) -> Series:
    by_step: dict[float, list[float]] = {}
    for path in paths:
        rows = read_csv_rows(path)
        running_total = 0.0
        for row in rows:
            step = to_float(row.get("step"))
            value = to_float(row.get(metric))
            if step is None or value is None:
                continue
            running_total += value
            by_step.setdefault(step, []).append(running_total)
    points = []
    for step in sorted(by_step.keys()):
        values = by_step[step]
        if values:
            points.append((step, sum(values) / len(values)))
    return Series(label, points, infer_color(label), "-", "o")


def first_step_at_threshold(path: str, metric: str, threshold: float) -> float | None:
    rows = read_csv_rows(path)
    for row in rows:
        step = to_float(row.get("step"))
        value = to_float(row.get(metric))
        if step is None or value is None:
            continue
        if value >= threshold:
            return step
    return None


def average_first_step_at_threshold(paths: list[str], metric: str, threshold: float) -> float | None:
    values = [first_step_at_threshold(path, metric, threshold) for path in paths]
    present = [float(value) for value in values if value is not None]
    return (sum(present) / len(present)) if present else None


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


def average_latest_training_rows(paths: list[str], label: str) -> dict[str, float | str | None]:
    rows = [load_latest_training_row(path, label) for path in paths]
    metrics = [
        "elapsed_wall_clock_seconds",
        "seconds_per_1k_timesteps",
        "train_fail",
        "train_timeout",
        "train_success",
    ]
    averaged: dict[str, float | str | None] = {"label": label}
    for metric in metrics:
        values = [row[metric] for row in rows if row.get(metric) is not None]
        averaged[metric] = (sum(float(v) for v in values) / len(values)) if values else None
    return averaged


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


def average_test_summaries(paths: list[str], label: str) -> dict[str, float | str | None]:
    rows = [load_test_summary(path, label) for path in paths]
    metrics = ["tsr", "dsr", "fail", "timeout", "mean_reward", "eval_wall_clock_seconds"]
    averaged: dict[str, float | str | None] = {"label": label}
    for metric in metrics:
        values = [row[metric] for row in rows if row.get(metric) is not None]
        averaged[metric] = (sum(float(v) for v in values) / len(values)) if values else None
    return averaged


def load_shared_test_summary(path: str, label: str | None = None) -> dict[str, float | str | None]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    summary = rows[-1]
    series_label = label or infer_method_label(path)
    return {
        "label": series_label,
        "joint_tsr": to_float(summary.get("joint_tsr")),
        "joint_dsr": to_float(summary.get("joint_dsr")),
        "car_1_tsr": to_float(summary.get("car_1_tsr")),
        "car_1_dsr": to_float(summary.get("car_1_dsr")),
        "car_2_tsr": to_float(summary.get("car_2_tsr")),
        "car_2_dsr": to_float(summary.get("car_2_dsr")),
        "fail": to_float(summary.get("fail") or summary.get("fail_episodes")),
        "timeout": to_float(summary.get("timeout") or summary.get("timeout_episodes")),
        "mean_reward": to_float(summary.get("mean_reward")),
        "eval_wall_clock_seconds": to_float(summary.get("eval_wall_clock_seconds")),
    }


def average_shared_test_summaries(paths: list[str], label: str) -> dict[str, float | str | None]:
    rows = [load_shared_test_summary(path, label) for path in paths]
    metrics = [
        "joint_tsr",
        "joint_dsr",
        "car_1_tsr",
        "car_1_dsr",
        "car_2_tsr",
        "car_2_dsr",
        "fail",
        "timeout",
        "mean_reward",
        "eval_wall_clock_seconds",
    ]
    averaged: dict[str, float | str | None] = {"label": label}
    for metric in metrics:
        values = [row[metric] for row in rows if row.get(metric) is not None]
        averaged[metric] = (sum(float(v) for v in values) / len(values)) if values else None
    return averaged


def save_line_plot(series_list: list[Series], title: str, ylabel: str, output_path: str) -> None:
    series_list = [series for series in series_list if series.points]
    if not series_list:
        return
    color_counts: dict[str, int] = {}
    for series in series_list:
        color_counts[series.color] = color_counts.get(series.color, 0) + 1
    variant_color_idx = 0
    adjusted_series: list[Series] = []
    for series in series_list:
        if color_counts.get(series.color, 0) > 1:
            adjusted_series.append(
                Series(
                    label=series.label,
                    points=series.points,
                    color=VARIANT_COLORS[variant_color_idx % len(VARIANT_COLORS)],
                    linestyle=series.linestyle,
                    marker=series.marker,
                )
            )
            variant_color_idx += 1
        else:
            adjusted_series.append(series)
    plt.figure(figsize=(10, 5))
    for series in adjusted_series:
        xs = [x for x, _ in series.points]
        ys = [y for _, y in series.points]
        plt.plot(
            xs,
            ys,
            marker=series.marker,
            linestyle=series.linestyle,
            linewidth=2,
            markersize=4,
            label=series.label,
            color=series.color,
        )
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


def build_convergence_rows(
    training_inputs: list[tuple[str | None, str]],
    avg_training_inputs: list[tuple[str, list[str]]],
    thresholds: list[float],
    metric: str = "tsr",
) -> list[dict[str, float | str | None]]:
    rows: list[dict[str, float | str | None]] = []
    if avg_training_inputs:
        for label, paths in avg_training_inputs:
            row: dict[str, float | str | None] = {"label": label}
            for threshold in thresholds:
                key = f"{metric}_steps_to_{threshold:.2f}"
                row[key] = average_first_step_at_threshold(paths, metric, threshold)
            rows.append(row)
        return rows

    for label, path in training_inputs:
        series_label = label or infer_method_label(path)
        row = {"label": series_label}
        for threshold in thresholds:
            key = f"{metric}_steps_to_{threshold:.2f}"
            row[key] = first_step_at_threshold(path, metric, threshold)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis result plots as PNG files using matplotlib.")
    parser.add_argument(
        "--training",
        action="append",
        help="Training CSV input as LABEL=path or just path. Can be passed multiple times. If omitted, thesis training defaults are auto-discovered.",
    )
    parser.add_argument(
        "--single-test",
        action="append",
        help="Single-agent final test CSV input as LABEL=path or just path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--shared-test",
        action="append",
        help="Shared-policy final test CSV input as LABEL=path or just path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--avg-training",
        action="append",
        help="Grouped training CSVs as LABEL=path1,path2,path3. Produces averaged method curves.",
    )
    parser.add_argument(
        "--avg-single-test",
        action="append",
        help="Grouped single-agent test CSVs as LABEL=path1,path2,path3. Produces averaged method summaries.",
    )
    parser.add_argument(
        "--avg-shared-test",
        action="append",
        help="Grouped shared-policy test CSVs as LABEL=path1,path2,path3. Produces averaged method summaries.",
    )
    parser.add_argument(
        "--output-dir",
        default="vis/thesis_results",
        help="Directory where plots and summaries are written.",
    )
    parser.add_argument(
        "--convergence-threshold",
        action="append",
        type=float,
        help="TSR threshold used for convergence summaries. Can be passed multiple times. Defaults to 0.70, 0.80, 0.90.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    convergence_thresholds = args.convergence_threshold or [0.70, 0.80, 0.90]

    training_inputs = parse_named_paths(args.training)
    avg_training_inputs = parse_named_groups(args.avg_training)
    if not training_inputs and not avg_training_inputs:
        training_inputs = autodiscover_training()

    if avg_training_inputs:
        tsr_series = [average_series(paths, "tsr", label) for label, paths in avg_training_inputs]
        dsr_series = [average_series(paths, "dsr", label) for label, paths in avg_training_inputs]
        reward_series = [average_series(paths, "mean_reward", label) for label, paths in avg_training_inputs]
        eval_fail_cumulative_series = [average_cumulative_series(paths, "fail", label) for label, paths in avg_training_inputs]
        train_fail_series = [average_series(paths, "train_fail", label) for label, paths in avg_training_inputs]
        train_fail_delta_series = [average_series(paths, "train_fail_since_last_eval", label) for label, paths in avg_training_inputs]
        runtime_rows = [average_latest_training_rows(paths, label) for label, paths in avg_training_inputs]
    else:
        tsr_series = [load_training_series(path, "tsr", label) for label, path in training_inputs]
        dsr_series = [load_training_series(path, "dsr", label) for label, path in training_inputs]
        reward_series = [load_training_series(path, "mean_reward", label) for label, path in training_inputs]
        eval_fail_cumulative_series = [load_cumulative_training_series(path, "fail", label) for label, path in training_inputs]
        train_fail_series = [load_training_series(path, "train_fail", label) for label, path in training_inputs]
        train_fail_delta_series = [load_training_series(path, "train_fail_since_last_eval", label) for label, path in training_inputs]
        runtime_rows = [load_latest_training_row(path, label) for label, path in training_inputs]

    save_line_plot(tsr_series, "TSR Learning Curve", "TSR", os.path.join(args.output_dir, "tsr_learning_curve.png"))
    save_line_plot(dsr_series, "DSR Learning Curve", "DSR", os.path.join(args.output_dir, "dsr_learning_curve.png"))
    save_line_plot(reward_series, "Mean Reward Learning Curve", "Mean reward", os.path.join(args.output_dir, "mean_reward_learning_curve.png"))
    save_line_plot(
        eval_fail_cumulative_series,
        "Cumulative Validation Failures",
        "cumulative eval fail",
        os.path.join(args.output_dir, "eval_fail_cumulative.png"),
    )
    save_line_plot(train_fail_series, "Cumulative Training Failures", "train_fail", os.path.join(args.output_dir, "train_fail_cumulative.png"))
    save_line_plot(train_fail_delta_series, "Training Failures Since Last Eval", "train_fail_since_last_eval", os.path.join(args.output_dir, "train_fail_since_last_eval.png"))
    save_grouped_bar_chart(
        runtime_rows,
        ["elapsed_wall_clock_seconds", "seconds_per_1k_timesteps"],
        "Training Runtime Overhead",
        os.path.join(args.output_dir, "runtime_overhead.png"),
    )
    save_summary_csv(os.path.join(args.output_dir, "training_runtime_summary.csv"), runtime_rows)
    convergence_rows = build_convergence_rows(training_inputs, avg_training_inputs, convergence_thresholds, metric="tsr")
    save_summary_csv(os.path.join(args.output_dir, "tsr_convergence_summary.csv"), convergence_rows)
    save_grouped_bar_chart(
        convergence_rows,
        [f"tsr_steps_to_{threshold:.2f}" for threshold in convergence_thresholds],
        "TSR Convergence Summary",
        os.path.join(args.output_dir, "tsr_convergence_summary.png"),
    )

    single_test_inputs = parse_named_paths(args.single_test)
    avg_single_test_inputs = parse_named_groups(args.avg_single_test)
    if avg_single_test_inputs:
        single_test_rows = [average_test_summaries(paths, label) for label, paths in avg_single_test_inputs]
        save_grouped_bar_chart(
            single_test_rows,
            ["tsr", "dsr", "fail", "timeout", "mean_reward"],
            "Final Single-Agent Test Summary (3-Seed Average)",
            os.path.join(args.output_dir, "single_agent_test_summary.png"),
        )
        save_summary_csv(os.path.join(args.output_dir, "single_agent_test_summary.csv"), single_test_rows)
    elif single_test_inputs:
        single_test_rows = [load_test_summary(path, label) for label, path in single_test_inputs]
        save_grouped_bar_chart(
            single_test_rows,
            ["tsr", "dsr", "fail", "timeout", "mean_reward"],
            "Final Single-Agent Test Summary",
            os.path.join(args.output_dir, "single_agent_test_summary.png"),
        )
        save_summary_csv(os.path.join(args.output_dir, "single_agent_test_summary.csv"), single_test_rows)

    shared_test_inputs = parse_named_paths(args.shared_test)
    avg_shared_test_inputs = parse_named_groups(args.avg_shared_test)
    if avg_shared_test_inputs:
        shared_test_rows = [average_shared_test_summaries(paths, label) for label, paths in avg_shared_test_inputs]
        save_grouped_bar_chart(
            shared_test_rows,
            ["joint_tsr", "joint_dsr", "car_1_tsr", "car_1_dsr", "car_2_tsr", "car_2_dsr", "fail", "timeout", "mean_reward"],
            "Final Shared-Policy Test Summary (3-Seed Average)",
            os.path.join(args.output_dir, "shared_policy_test_summary.png"),
        )
        save_summary_csv(os.path.join(args.output_dir, "shared_policy_test_summary.csv"), shared_test_rows)
    elif shared_test_inputs:
        shared_test_rows = [load_shared_test_summary(path, label) for label, path in shared_test_inputs]
        save_grouped_bar_chart(
            shared_test_rows,
            ["joint_tsr", "joint_dsr", "car_1_tsr", "car_1_dsr", "car_2_tsr", "car_2_dsr", "fail", "timeout", "mean_reward"],
            "Final Shared-Policy Test Summary",
            os.path.join(args.output_dir, "shared_policy_test_summary.png"),
        )
        save_summary_csv(os.path.join(args.output_dir, "shared_policy_test_summary.csv"), shared_test_rows)

    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
