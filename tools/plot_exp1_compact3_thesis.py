import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAIN_COLORS = {
    "tsr": "#1f77b4",
    "timeout_rate": "#e45756",
    "ratio": "#4c78a8",
    "l1": "#72b7b2",
    "rollout_tsr": "#54a24b",
    "throughput": "#f58518",
}
ACTION_COLORS = {
    "slow": "#9c755f",
    "normal": "#72b7b2",
    "fast": "#54a24b",
    "stop": "#e45756",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot thesis-ready Experiment 1 compact3 figures."
    )
    parser.add_argument("--test-csvs", nargs="+", required=True, help="Fixed-test full metrics CSVs, one per K.")
    parser.add_argument(
        "--rollout-detail-csvs",
        nargs="+",
        required=True,
        help="Continuous rollout detailed metrics CSVs, one per K.",
    )
    parser.add_argument(
        "--train-eval-csvs",
        nargs="+",
        required=True,
        help="Training eval metrics CSVs, one per K.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "plots" / "exp1_compact3_thesis"),
        help="Directory where figures and summary TSV will be written.",
    )
    parser.add_argument(
        "--title-prefix",
        default="Experiment 1",
        help="Figure title prefix.",
    )
    return parser.parse_args()


def ensure_dir(path_str):
    Path(path_str).mkdir(parents=True, exist_ok=True)


def read_csv_rows(path):
    with open(path, "r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def read_latest_row(path):
    rows = read_csv_rows(path)
    row = max(
        rows,
        key=lambda r: (
            to_float(r.get("timestep"), 0.0),
            to_float(r.get("eval_index"), 0.0),
        ),
    )
    row["_source_path"] = str(path)
    return row


def to_float(value, default=0.0):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def infer_k(row):
    return int(to_float(row.get("k", row.get("num_rl_agents", 0)), 0))


def latest_train_row(rows):
    return max(rows, key=lambda row: to_float(row.get("timestep"), 0.0))


def build_maps(test_rows, rollout_rows, train_eval_rows):
    test_by_k = {infer_k(row): row for row in test_rows}
    rollout_by_k = {infer_k(row): row for row in rollout_rows}
    train_by_k = {}
    for rows in train_eval_rows:
        k_value = infer_k(rows[0])
        train_by_k[k_value] = rows
    return test_by_k, rollout_by_k, train_by_k


def action_fractions(row):
    counts = {
        action: to_float(row.get(f"action_{action}_count"), 0.0)
        for action in ["slow", "normal", "fast", "stop"]
    }
    total = sum(counts.values())
    if total <= 0:
        return {action: 0.0 for action in counts}
    return {action: counts[action] / total for action in counts}


def write_summary_tsv(test_by_k, rollout_by_k, train_by_k, output_dir):
    output_path = Path(output_dir) / "exp1_compact3_summary.tsv"
    fields = [
        "k",
        "test_tsr",
        "test_timeout_rate",
        "test_mean_policy_to_expert_step_ratio",
        "test_mean_l1_shift_base_to_shielded",
        "rollout_micro_rollout_tsr",
        "rollout_successful_goals_per_1k_steps",
        "train_latest_rollout_failures_since_last_eval",
        "train_latest_rollout_timeouts_since_last_eval",
    ]
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for k_value in sorted(test_by_k.keys()):
            latest_train = latest_train_row(train_by_k[k_value]) if k_value in train_by_k else {}
            writer.writerow(
                {
                    "k": k_value,
                    "test_tsr": f"{to_float(test_by_k[k_value].get('tsr')):.4f}",
                    "test_timeout_rate": f"{to_float(test_by_k[k_value].get('timeout_rate')):.4f}",
                    "test_mean_policy_to_expert_step_ratio": f"{to_float(test_by_k[k_value].get('mean_policy_to_expert_step_ratio')):.4f}",
                    "test_mean_l1_shift_base_to_shielded": f"{to_float(test_by_k[k_value].get('mean_l1_shift_base_to_shielded')):.4f}",
                    "rollout_micro_rollout_tsr": f"{to_float(rollout_by_k[k_value].get('micro_rollout_tsr')):.4f}",
                    "rollout_successful_goals_per_1k_steps": f"{to_float(rollout_by_k[k_value].get('successful_goals_per_1k_steps')):.4f}",
                    "train_latest_rollout_failures_since_last_eval": int(to_float(latest_train.get("rollout_failures_since_last_eval"), 0.0)),
                    "train_latest_rollout_timeouts_since_last_eval": int(to_float(latest_train.get("rollout_timeouts_since_last_eval"), 0.0)),
                }
            )


def make_main_summary_table(test_by_k, rollout_by_k, output_dir, title_prefix):
    ks = sorted(test_by_k.keys())
    headers = [
        "K",
        "Test TSR",
        "Test Timeout",
        "Step Ratio",
        "L1 Shift",
        "Rollout TSR",
        "Goals / 1k",
    ]
    cell_text = []
    for k_value in ks:
        test_row = test_by_k[k_value]
        rollout_row = rollout_by_k[k_value]
        cell_text.append(
            [
                str(k_value),
                f"{to_float(test_row.get('tsr')):.2f}",
                f"{to_float(test_row.get('timeout_rate')):.2f}",
                f"{to_float(test_row.get('mean_policy_to_expert_step_ratio')):.2f}",
                f"{to_float(test_row.get('mean_l1_shift_base_to_shielded')):.2f}",
                f"{to_float(rollout_row.get('micro_rollout_tsr')):.2f}",
                f"{to_float(rollout_row.get('successful_goals_per_1k_steps')):.1f}",
            ]
        )

    fig, ax = plt.subplots(figsize=(10.5, 2.2 + 0.35 * max(len(ks), 1)))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#e8eef7")
            cell.set_text_props(weight="bold")
    ax.set_title(f"{title_prefix}: Main Metrics", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "exp1_main_summary_table.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_main_results_figure(test_by_k, rollout_by_k, output_dir, title_prefix):
    ks = sorted(test_by_k.keys())
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4))
    specs = [
        ("tsr", "Fixed Test TSR", lambda k: to_float(test_by_k[k].get("tsr")), MAIN_COLORS["tsr"], (0.0, 1.05)),
        ("timeout_rate", "Fixed Test Timeout Rate", lambda k: to_float(test_by_k[k].get("timeout_rate")), MAIN_COLORS["timeout_rate"], (0.0, 1.05)),
        ("ratio", "Fixed Test Step Ratio to Expert", lambda k: to_float(test_by_k[k].get("mean_policy_to_expert_step_ratio")), MAIN_COLORS["ratio"], None),
        ("l1", "Fixed Test Mean L1 Shift", lambda k: to_float(test_by_k[k].get("mean_l1_shift_base_to_shielded")), MAIN_COLORS["l1"], None),
        ("rollout_tsr", "Rollout Micro TSR", lambda k: to_float(rollout_by_k[k].get("micro_rollout_tsr")), MAIN_COLORS["rollout_tsr"], (0.0, 1.05)),
        ("throughput", "Rollout Successful Goals per 1k Steps", lambda k: to_float(rollout_by_k[k].get("successful_goals_per_1k_steps")), MAIN_COLORS["throughput"], None),
    ]
    for ax, (_key, title, getter, color, ylim) in zip(axes.flat, specs):
        ys = [getter(k) for k in ks]
        ax.plot(ks, ys, marker="o", linewidth=2.2, color=color)
        ax.set_title(title)
        ax.set_xlabel("K")
        ax.set_xticks(ks)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
    fig.suptitle(f"{title_prefix}: Main Results", y=1.02)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "exp1_main_results.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_training_diagnostic_figure(train_by_k, output_dir, title_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), sharex=False)
    for k_value in sorted(train_by_k.keys()):
        rows = train_by_k[k_value]
        xs = [to_float(row.get("timestep")) for row in rows]
        failures = [to_float(row.get("rollout_failures_since_last_eval")) for row in rows]
        timeouts = [to_float(row.get("rollout_timeouts_since_last_eval")) for row in rows]
        axes[0].plot(xs, failures, marker="o", linewidth=2.0, label=f"K={k_value}")
        axes[1].plot(xs, timeouts, marker="o", linewidth=2.0, label=f"K={k_value}")
    axes[0].set_title("Training Rollout Failures Since Last Eval")
    axes[1].set_title("Training Rollout Timeouts Since Last Eval")
    for ax in axes:
        ax.set_xlabel("Training Timestep")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle(f"{title_prefix}: Training Diagnostics", y=1.02)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "exp1_training_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_behavior_figure(test_by_k, output_dir, title_prefix):
    ks = sorted(test_by_k.keys())
    fractions_by_k = {k: action_fractions(row) for k, row in test_by_k.items()}
    action_names = ["slow", "normal", "fast", "stop"]
    action_names = [
        action_name
        for action_name in action_names
        if any(fractions_by_k[k][action_name] > 0.0 for k in ks)
    ]
    if not action_names:
        action_names = ["normal", "fast", "stop"]
    xs = np.arange(len(ks))
    bottoms = np.zeros(len(ks), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for action_name in action_names:
        heights = np.array([fractions_by_k[k][action_name] for k in ks], dtype=np.float64)
        ax.bar(
            xs,
            heights,
            bottom=bottoms,
            color=ACTION_COLORS[action_name],
            label=action_name.capitalize(),
            width=0.7,
        )
        bottoms += heights
    ax.set_xticks(xs, [str(k) for k in ks])
    ax.set_xlabel("K")
    ax.set_ylabel("Action Fraction")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title(f"{title_prefix}: Fixed-Test Action Profile")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "exp1_behavior_profile.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    test_rows = [read_latest_row(Path(path)) for path in args.test_csvs]
    rollout_rows = [read_latest_row(Path(path)) for path in args.rollout_detail_csvs]
    train_eval_rows = [read_csv_rows(Path(path)) for path in args.train_eval_csvs]

    test_by_k, rollout_by_k, train_by_k = build_maps(test_rows, rollout_rows, train_eval_rows)
    write_summary_tsv(test_by_k, rollout_by_k, train_by_k, args.output_dir)
    make_main_summary_table(test_by_k, rollout_by_k, args.output_dir, args.title_prefix)
    make_main_results_figure(test_by_k, rollout_by_k, args.output_dir, args.title_prefix)
    make_training_diagnostic_figure(train_by_k, args.output_dir, args.title_prefix)
    make_behavior_figure(test_by_k, args.output_dir, args.title_prefix)


if __name__ == "__main__":
    main()
