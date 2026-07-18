import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

TEST_COLOR = "#1f77b4"
ROLLOUT_COLOR = "#d62728"
ACTION_COLORS = {
    "slow": "#4c78a8",
    "normal": "#72b7b2",
    "fast": "#54a24b",
    "stop": "#e45756",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Experiment 1 dense PLS results from selected fixed-test and rollout CSVs."
    )
    parser.add_argument(
        "--test-csvs",
        nargs="+",
        required=True,
        help="Chosen fixed-test metrics CSVs, one per K checkpoint selection.",
    )
    parser.add_argument(
        "--rollout-csvs",
        nargs="*",
        default=[],
        help="Chosen rollout compact metrics CSVs, one per K checkpoint selection.",
    )
    parser.add_argument(
        "--rollout-detail-csvs",
        nargs="*",
        default=[],
        help="Chosen rollout detailed continuous metrics CSVs, one per K checkpoint selection.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "plots" / "exp1_dense_results"),
        help="Directory where the summary tables and figures will be written.",
    )
    parser.add_argument(
        "--title-prefix",
        default="Experiment 1",
        help="Prefix used in figure titles.",
    )
    return parser.parse_args()


def read_single_row(csv_path):
    with open(csv_path, "r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {csv_path}, found {len(rows)}")
    row = rows[0]
    row["_source_path"] = str(csv_path)
    return row


def to_float(value, default=0.0):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def infer_k(row):
    if row.get("k") not in (None, ""):
        return int(to_float(row.get("k"), 0))
    return int(to_float(row.get("num_rl_agents"), 0))


def infer_checkpoint_label(row):
    exp_name = str(row.get("exp_name", ""))
    parts = exp_name.split("_")
    if len(parts) > 0:
        tail = parts[-1]
        if tail.isdigit():
            return tail
    path_text = str(row.get("_source_path", ""))
    for token in reversed(path_text.replace("\\", "/").split("/")):
        if token.endswith(".csv"):
            stem = token[:-4]
            for segment in reversed(stem.split("_")):
                if segment.isdigit():
                    return segment
    return ""


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_action_fractions(row, prefix="action"):
    counts = {
        action: to_float(row.get(f"{prefix}_{action}_count"), 0.0)
        for action in ["slow", "normal", "fast", "stop"]
    }
    total = sum(counts.values())
    if total <= 0:
        return {action: 0.0 for action in counts}
    return {action: counts[action] / total for action in counts}


def build_test_map(rows):
    data = {}
    for row in rows:
        data[infer_k(row)] = row
    return data


def build_rollout_map(compact_rows, detailed_rows):
    compact_by_k = {infer_k(row): row for row in compact_rows}
    detailed_by_k = {infer_k(row): row for row in detailed_rows}
    merged = {}
    for k_value in sorted(set(compact_by_k.keys()) | set(detailed_by_k.keys())):
        merged[k_value] = {
            "compact": compact_by_k.get(k_value),
            "detail": detailed_by_k.get(k_value),
        }
    return merged


def write_summary_tsv(test_by_k, rollout_by_k, output_dir):
    output_path = Path(output_dir) / "exp1_summary.tsv"
    fields = [
        "k",
        "test_checkpoint",
        "test_tsr",
        "test_timeout_rate",
        "test_mean_episode_length",
        "test_mean_agent_tsr",
        "rollout_checkpoint",
        "rollout_micro_tsr",
        "rollout_failure_rate",
        "rollout_timeout_rate",
        "rollout_mean_steps_per_resolved_goal_attempt",
        "rollout_hazard_step_rate",
        "rollout_mean_safety_gain",
        "rollout_mean_l1_shift",
    ]
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for k_value in sorted(set(test_by_k.keys()) | set(rollout_by_k.keys())):
            test_row = test_by_k.get(k_value)
            rollout_pack = rollout_by_k.get(k_value, {})
            rollout_compact = rollout_pack.get("compact")
            rollout_detail = rollout_pack.get("detail")
            writer.writerow(
                {
                    "k": k_value,
                    "test_checkpoint": infer_checkpoint_label(test_row or {}),
                    "test_tsr": f"{to_float((test_row or {}).get('tsr')):.4f}" if test_row else "",
                    "test_timeout_rate": f"{to_float((test_row or {}).get('timeout_rate')):.4f}" if test_row else "",
                    "test_mean_episode_length": f"{to_float((test_row or {}).get('mean_episode_length')):.4f}" if test_row else "",
                    "test_mean_agent_tsr": f"{to_float((test_row or {}).get('mean_agent_tsr')):.4f}" if test_row else "",
                    "rollout_checkpoint": infer_checkpoint_label((rollout_compact or rollout_detail) or {}),
                    "rollout_micro_tsr": f"{to_float((rollout_compact or {}).get('micro_rollout_tsr')):.4f}" if rollout_compact else "",
                    "rollout_failure_rate": f"{to_float((rollout_compact or {}).get('micro_rollout_failure_rate')):.4f}" if rollout_compact else "",
                    "rollout_timeout_rate": f"{to_float((rollout_compact or rollout_detail or {}).get('micro_rollout_timeout_rate')):.4f}" if (rollout_compact or rollout_detail) else "",
                    "rollout_mean_steps_per_resolved_goal_attempt": f"{to_float((rollout_compact or {}).get('mean_steps_per_resolved_goal_attempt')):.4f}" if rollout_compact else "",
                    "rollout_hazard_step_rate": f"{to_float((rollout_detail or {}).get('hazard_step_rate')):.4f}" if rollout_detail else "",
                    "rollout_mean_safety_gain": f"{to_float((rollout_detail or {}).get('mean_safety_gain')):.4f}" if rollout_detail else "",
                    "rollout_mean_l1_shift": f"{to_float((rollout_detail or {}).get('mean_l1_shift_base_to_shielded')):.4f}" if rollout_detail else "",
                }
            )


def make_summary_table_figure(test_by_k, rollout_by_k, output_dir, title_prefix):
    ks = sorted(set(test_by_k.keys()) | set(rollout_by_k.keys()))
    headers = [
        "K",
        "Test TSR",
        "Test Timeout",
        "Test Mean Len",
        "Rollout TSR",
        "Rollout Timeout",
        "Rollout Steps/Goal",
    ]
    cell_text = []
    for k_value in ks:
        test_row = test_by_k.get(k_value, {})
        rollout_compact = rollout_by_k.get(k_value, {}).get("compact", {})
        rollout_detail = rollout_by_k.get(k_value, {}).get("detail", {})
        timeout_value = rollout_detail.get("micro_rollout_timeout_rate", rollout_compact.get("micro_rollout_timeout_rate", ""))
        cell_text.append(
            [
                str(k_value),
                f"{to_float(test_row.get('tsr')):.2f}" if test_row else "-",
                f"{to_float(test_row.get('timeout_rate')):.2f}" if test_row else "-",
                f"{to_float(test_row.get('mean_episode_length')):.1f}" if test_row else "-",
                f"{to_float(rollout_compact.get('micro_rollout_tsr')):.2f}" if rollout_compact else "-",
                f"{to_float(timeout_value):.2f}" if rollout_compact or rollout_detail else "-",
                f"{to_float(rollout_compact.get('mean_steps_per_resolved_goal_attempt')):.1f}" if rollout_compact else "-",
            ]
        )

    fig, ax = plt.subplots(figsize=(10.5, 2.2 + 0.35 * max(len(ks), 1)))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#e8eef7")
            cell.set_text_props(weight="bold")
    ax.set_title(f"{title_prefix}: Main Summary", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "exp1_main_summary_table.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_scaling_figure(test_by_k, rollout_by_k, output_dir, title_prefix):
    ks = sorted(set(test_by_k.keys()) | set(rollout_by_k.keys()))
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))

    test_tsr = [to_float(test_by_k.get(k, {}).get("tsr")) for k in ks if k in test_by_k]
    test_tsr_x = [k for k in ks if k in test_by_k]
    rollout_tsr = [to_float(rollout_by_k.get(k, {}).get("compact", {}).get("micro_rollout_tsr")) for k in ks if rollout_by_k.get(k, {}).get("compact")]
    rollout_tsr_x = [k for k in ks if rollout_by_k.get(k, {}).get("compact")]
    axes[0].plot(test_tsr_x, test_tsr, marker="o", linewidth=2.2, color=TEST_COLOR, label="Fixed Test TSR")
    axes[0].plot(rollout_tsr_x, rollout_tsr, marker="o", linewidth=2.2, color=ROLLOUT_COLOR, label="Rollout Micro TSR")
    axes[0].set_xlabel("Number of RL Agents (K)")
    axes[0].set_ylabel("Success Rate")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xticks(ks)
    axes[0].set_title("Task Completion")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    test_len = [to_float(test_by_k.get(k, {}).get("mean_episode_length")) for k in ks if k in test_by_k]
    rollout_steps = [to_float(rollout_by_k.get(k, {}).get("compact", {}).get("mean_steps_per_resolved_goal_attempt")) for k in ks if rollout_by_k.get(k, {}).get("compact")]
    axes[1].plot(test_tsr_x, test_len, marker="o", linewidth=2.2, color=TEST_COLOR, label="Fixed Test Mean Episode Length")
    axes[1].plot(rollout_tsr_x, rollout_steps, marker="o", linewidth=2.2, color=ROLLOUT_COLOR, label="Rollout Steps per Resolved Goal")
    axes[1].set_xlabel("Number of RL Agents (K)")
    axes[1].set_ylabel("Steps")
    axes[1].set_xticks(ks)
    axes[1].set_title("Efficiency")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle(f"{title_prefix}: Scaling of Success and Efficiency", y=1.02)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "exp1_scaling_main.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_shield_dependence_figure(test_by_k, rollout_by_k, output_dir, title_prefix):
    ks = sorted(set(test_by_k.keys()) | set(rollout_by_k.keys()))
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharex=True)
    metric_specs = [
        ("shield_intervention_rate", "Shield Intervention Rate"),
        ("mean_safety_gain", "Mean Safety Gain"),
        ("mean_l1_shift_base_to_shielded", "Mean L1 Shift"),
    ]
    for ax, (metric_key, metric_label) in zip(axes, metric_specs):
        test_vals = [to_float(test_by_k.get(k, {}).get(metric_key)) for k in ks if k in test_by_k]
        test_x = [k for k in ks if k in test_by_k]
        rollout_vals = [to_float(rollout_by_k.get(k, {}).get("detail", {}).get(metric_key)) for k in ks if rollout_by_k.get(k, {}).get("detail")]
        rollout_x = [k for k in ks if rollout_by_k.get(k, {}).get("detail")]
        ax.plot(test_x, test_vals, marker="o", linewidth=2.0, color=TEST_COLOR, label="Fixed Test")
        ax.plot(rollout_x, rollout_vals, marker="o", linewidth=2.0, color=ROLLOUT_COLOR, label="Rollout")
        ax.set_title(metric_label)
        ax.set_xlabel("K")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(f"{title_prefix}: Shield Dependence", y=1.03)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(Path(output_dir) / "exp1_shield_dependence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_behavior_figure(test_by_k, rollout_by_k, output_dir, title_prefix):
    ks = sorted(set(test_by_k.keys()) | set(rollout_by_k.keys()))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
    sources = [
        ("Fixed Test", {k: normalize_action_fractions(row) for k, row in test_by_k.items()}),
        (
            "Rollout",
            {
                k: normalize_action_fractions(rollout_by_k[k]["detail"])
                for k in ks
                if rollout_by_k.get(k, {}).get("detail") is not None
            },
        ),
    ]
    for ax, (title, fractions_by_k) in zip(axes, sources):
        xs = np.arange(len(ks))
        bottoms = np.zeros(len(ks), dtype=np.float64)
        for action_name in ["slow", "normal", "fast", "stop"]:
            heights = np.array(
                [fractions_by_k.get(k, {}).get(action_name, 0.0) for k in ks],
                dtype=np.float64,
            )
            ax.bar(
                xs,
                heights,
                bottom=bottoms,
                color=ACTION_COLORS[action_name],
                label=action_name.capitalize(),
                width=0.7,
            )
            bottoms += heights
        ax.set_title(title)
        ax.set_xticks(xs, [str(k) for k in ks])
        ax.set_xlabel("K")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Action Fraction")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"{title_prefix}: Shielded Action Profile", y=1.03)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(Path(output_dir) / "exp1_behavior_stacked.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_supporting_diagnostics_figure(test_by_k, rollout_by_k, output_dir, title_prefix):
    ks = sorted(set(test_by_k.keys()) | set(rollout_by_k.keys()))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))

    test_timeout = [to_float(test_by_k.get(k, {}).get("timeout_rate")) for k in ks if k in test_by_k]
    test_timeout_x = [k for k in ks if k in test_by_k]
    rollout_timeout = [to_float(rollout_by_k.get(k, {}).get("detail", {}).get("micro_rollout_timeout_rate")) for k in ks if rollout_by_k.get(k, {}).get("detail")]
    rollout_timeout_x = [k for k in ks if rollout_by_k.get(k, {}).get("detail")]
    rollout_fail = [to_float(rollout_by_k.get(k, {}).get("detail", {}).get("micro_rollout_failure_rate")) for k in ks if rollout_by_k.get(k, {}).get("detail")]

    axes[0].plot(test_timeout_x, test_timeout, marker="o", linewidth=2.1, color=TEST_COLOR, label="Fixed Test Timeout Rate")
    axes[0].plot(rollout_timeout_x, rollout_timeout, marker="o", linewidth=2.1, color=ROLLOUT_COLOR, label="Rollout Timeout Rate")
    axes[0].plot(rollout_timeout_x, rollout_fail, marker="o", linewidth=2.1, color="#9467bd", label="Rollout Failure Rate")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Rate")
    axes[0].set_xticks(ks)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Timeout / Failure Exposure")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    test_hazard = [to_float(test_by_k.get(k, {}).get("hazard_step_rate")) for k in ks if k in test_by_k]
    rollout_hazard = [to_float(rollout_by_k.get(k, {}).get("detail", {}).get("hazard_step_rate")) for k in ks if rollout_by_k.get(k, {}).get("detail")]
    test_forced = [to_float(test_by_k.get(k, {}).get("shield_forced_stop_rate")) for k in ks if k in test_by_k]
    rollout_forced = [to_float(rollout_by_k.get(k, {}).get("detail", {}).get("shield_forced_stop_rate")) for k in ks if rollout_by_k.get(k, {}).get("detail")]

    axes[1].plot(test_timeout_x, test_hazard, marker="o", linewidth=2.1, color=TEST_COLOR, label="Fixed Test Hazard Step Rate")
    axes[1].plot(rollout_timeout_x, rollout_hazard, marker="o", linewidth=2.1, color=ROLLOUT_COLOR, label="Rollout Hazard Step Rate")
    axes[1].plot(test_timeout_x, test_forced, marker="o", linewidth=1.8, linestyle="--", color="#2ca02c", label="Fixed Test Forced-Stop Rate")
    axes[1].plot(rollout_timeout_x, rollout_forced, marker="o", linewidth=1.8, linestyle="--", color="#8c564b", label="Rollout Forced-Stop Rate")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Rate")
    axes[1].set_xticks(ks)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Hazard / Forced-Stop Diagnostics")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle(f"{title_prefix}: Supporting Diagnostics", y=1.03)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "exp1_supporting_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    test_rows = [read_single_row(Path(path)) for path in args.test_csvs]
    rollout_compact_rows = [read_single_row(Path(path)) for path in args.rollout_csvs]
    rollout_detail_rows = [read_single_row(Path(path)) for path in args.rollout_detail_csvs]

    test_by_k = build_test_map(test_rows)
    rollout_by_k = build_rollout_map(rollout_compact_rows, rollout_detail_rows)

    write_summary_tsv(test_by_k, rollout_by_k, args.output_dir)
    make_summary_table_figure(test_by_k, rollout_by_k, args.output_dir, args.title_prefix)
    make_scaling_figure(test_by_k, rollout_by_k, args.output_dir, args.title_prefix)
    make_shield_dependence_figure(test_by_k, rollout_by_k, args.output_dir, args.title_prefix)
    make_behavior_figure(test_by_k, rollout_by_k, args.output_dir, args.title_prefix)
    make_supporting_diagnostics_figure(test_by_k, rollout_by_k, args.output_dir, args.title_prefix)


if __name__ == "__main__":
    main()
