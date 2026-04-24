import argparse
import csv
import os

import matplotlib.pyplot as plt


METHOD_ORDER = ["PPO", "PPO + DLS", "PPO + PLS"]
DISPLAY_LABELS = {
    "PPO": "PPO",
    "PPO + DLS": "DLS",
    "PPO + PLS": "PLS",
}
COLORS = {
    "single": "#4C78A8",
    "multi": "#E45756",
}


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    return float(value)


def load_metric_map(path, label_col, metric_col):
    data = {}
    for row in read_csv_rows(path):
        label = row.get(label_col, "").strip()
        metric = to_float(row.get(metric_col))
        if label and metric is not None:
            data[label] = metric
    return data


def build_order(single_map, multi_map):
    labels = []
    for method in METHOD_ORDER:
        if method in single_map and method in multi_map:
            labels.append(method)
    for method in sorted(set(single_map) & set(multi_map)):
        if method not in labels:
            labels.append(method)
    if not labels:
        raise ValueError("No overlapping method labels found between the two CSV files.")
    return labels


def add_bar_labels(ax, values, positions, side):
    for value, pos in zip(values, positions):
        text = f"{abs(value) * 100:.1f}%"
        if side == "left":
            ax.text(value - 0.025, pos, text, va="center", ha="right", fontsize=10)
        else:
            ax.text(value + 0.025, pos, text, va="center", ha="left", fontsize=10)


def plot_chart(single_csv, multi_csv, output_path, title):
    single_map = load_metric_map(single_csv, "label", "tsr")
    multi_map = load_metric_map(multi_csv, "label", "joint_tsr")
    labels = build_order(single_map, multi_map)

    single_values = [-single_map[label] for label in labels]
    multi_values = [multi_map[label] for label in labels]
    positions = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.barh(positions, single_values, height=0.62, color=COLORS["single"], label="Single-Agent TSR")
    ax.barh(positions, multi_values, height=0.62, color=COLORS["multi"], label="Multi-Agent TSR")

    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_yticks(positions)
    ax.set_yticklabels([DISPLAY_LABELS.get(label, label) for label in labels], fontsize=11)
    ax.invert_yaxis()

    max_value = max(max(abs(v) for v in single_values), max(abs(v) for v in multi_values))
    limit = min(1.15, max(0.75, max_value + 0.12))
    ax.set_xlim(-limit, limit)

    tick_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    tick_values = [v for v in tick_values if -limit <= v <= limit]
    ax.set_xticks(tick_values)
    ax.set_xticklabels([f"{abs(v) * 100:.0f}%" if v != 0 else "0%" for v in tick_values])
    ax.set_xlabel("TSR", fontsize=11)
    ax.set_title(title, fontsize=14, pad=12)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.18))

    add_bar_labels(ax, single_values, positions, side="left")
    add_bar_labels(ax, multi_values, positions, side="right")

    fig.text(0.26, 0.02, "Single-Agent", ha="center", fontsize=10)
    fig.text(0.74, 0.02, "Multi-Agent", ha="center", fontsize=10)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot a two-sided TSR comparison bar chart.")
    parser.add_argument(
        "--single_csv",
        default="vis/thesis_results_test_avg_3seeds/single_agent_test_summary.csv",
        help="CSV containing per-method single-agent TSR values.",
    )
    parser.add_argument(
        "--multi_csv",
        default="vis/thesis_results_test_avg_3seeds/shared_policy_test_summary.csv",
        help="CSV containing per-method shared-policy TSR values.",
    )
    parser.add_argument(
        "--output",
        default="vis/thesis_results_test_avg_3seeds/two_sided_tsr_barchart.png",
        help="Path to the output image.",
    )
    parser.add_argument(
        "--title",
        default="Single-Agent vs Multi-Agent TSR",
        help="Plot title.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_chart(args.single_csv, args.multi_csv, args.output, args.title)
