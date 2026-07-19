import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Experiment 1 TSR bar chart from direct result folders.")
    parser.add_argument(
        "--results-dir",
        default=str(ROOT / "results" / "experiment_family_1"),
        help="Base results directory containing ppo_k* and plpg_k* folders.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "thesis" / "figures" / "exp1_perfect_tsr_barchart.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--ppo-prefix",
        default="ppo",
        help="Folder prefix for PPO results, without the trailing _k{K}.",
    )
    parser.add_argument(
        "--plpg-prefix",
        default="plpg",
        help="Folder prefix for PLPG results, without the trailing _k{K}.",
    )
    parser.add_argument(
        "--title",
        default="Experiment 1: TSR under Perfect Symbolic Observations",
        help="Plot title.",
    )
    return parser.parse_args()


def read_single_row(csv_path):
    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {csv_path}, found {len(rows)}")
    return rows[0]


def load_tsr(results_dir, model_prefix, k_value):
    folder = Path(results_dir) / f"{model_prefix}_k{k_value}"
    matches = sorted(folder.glob("*_test_metrics.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one test metrics CSV in {folder}, found {len(matches)}")
    row = read_single_row(matches[0])
    return float(row["tsr"])


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ks = [1, 2, 3]
    ppo_tsrs = [load_tsr(args.results_dir, args.ppo_prefix, k) for k in ks]
    plpg_tsrs = [load_tsr(args.results_dir, args.plpg_prefix, k) for k in ks]

    x = np.arange(len(ks))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar(x - width / 2, ppo_tsrs, width, label="PPO", color="#1f77b4")
    ax.bar(x + width / 2, plpg_tsrs, width, label="PLPG", color="#d62728")

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Number of RL Agents (K)")
    ax.set_ylabel("TSR")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(args.title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
