import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot thesis test results from Family 1 and Family 2 CSVs.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "plots" / "thesis_test_results"),
        help="Directory where plots will be written.",
    )
    return parser.parse_args()


def read_rows(csv_paths):
    rows = []
    for path in csv_paths:
        with open(path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["_source_path"] = str(path)
                rows.append(row)
    return rows


def infer_variant(row):
    text = " ".join(
        [
            str(row.get("exp_name", "")),
            str(row.get("_source_path", "")),
        ]
    ).lower()
    if "plpg_fine" in text:
        return "PLPG Fine"
    if "plpg" in text:
        return "PLPG"
    if "ppo" in text:
        return "PPO"
    return str(row.get("method", "Unknown"))


def infer_difficulty(row):
    return str(row.get("difficulty", "unknown")).strip().lower()


def infer_k(row):
    return int(float(row.get("num_rl_agents", 0)))


def to_float(value):
    if value in (None, ""):
        return 0.0
    return float(value)


def collect_test_csvs():
    family1 = sorted((ROOT / "results" / "experiment_family_1").rglob("*_test_metrics.csv"))
    family2 = sorted((ROOT / "results" / "experiment_family_2").rglob("*_test_metrics.csv"))
    return family1 + family2


def build_summary(rows):
    summary = {}
    for row in rows:
        difficulty = infer_difficulty(row)
        variant = infer_variant(row)
        k_value = infer_k(row)
        summary[(difficulty, variant, k_value)] = {
            "tsr": to_float(row.get("tsr")),
            "failure_rate": to_float(row.get("failure_rate")),
            "timeout_rate": to_float(row.get("timeout_rate")),
            "mean_reward": to_float(row.get("mean_reward")),
        }
    return summary


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_family1_tsr(summary, output_dir):
    difficulty = "easy"
    variants = ["PPO", "PLPG", "PLPG Fine"]
    ks = [1, 2, 3]

    plt.figure(figsize=(7, 4.5))
    for variant in variants:
        ys = [summary[(difficulty, variant, k)]["tsr"] for k in ks if (difficulty, variant, k) in summary]
        xs = [k for k in ks if (difficulty, variant, k) in summary]
        if xs:
            plt.plot(xs, ys, marker="o", linewidth=2, label=variant)

    plt.xticks(ks)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Number of RL Agents (K)")
    plt.ylabel("TSR")
    plt.title("Family 1: Easy Multi-Agent Benchmark")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "family1_easy_tsr.png", dpi=200)
    plt.close()


def plot_family2_tsr(summary, output_dir):
    difficulties = ["easy", "medium", "hard"]
    variants = ["PPO", "PLPG", "PLPG Fine"]
    ks = [1, 2, 3]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, difficulty in zip(axes, difficulties):
        for variant in variants:
            ys = [summary[(difficulty, variant, k)]["tsr"] for k in ks if (difficulty, variant, k) in summary]
            xs = [k for k in ks if (difficulty, variant, k) in summary]
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=2, label=variant)
        ax.set_title(difficulty.capitalize())
        ax.set_xticks(ks)
        ax.set_xlabel("K")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("TSR")
    axes[0].set_ylim(0.0, 1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("TSR Across Difficulty and RL-Agent Count", y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(Path(output_dir) / "family2_difficulty_tsr.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary_table(summary, output_dir):
    output_path = Path(output_dir) / "test_summary.tsv"
    rows = []
    for difficulty, variant, k_value in sorted(summary.keys()):
        metrics = summary[(difficulty, variant, k_value)]
        rows.append(
            [
                difficulty,
                variant,
                str(k_value),
                f"{metrics['tsr']:.4f}",
                f"{metrics['failure_rate']:.4f}",
                f"{metrics['timeout_rate']:.4f}",
                f"{metrics['mean_reward']:.4f}",
            ]
        )

    with open(output_path, "w", newline="") as handle:
        handle.write("difficulty\tvariant\tk\ttsr\tfailure_rate\ttimeout_rate\tmean_reward\n")
        for row in rows:
            handle.write("\t".join(row) + "\n")


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    csv_paths = collect_test_csvs()
    rows = read_rows(csv_paths)
    summary = build_summary(rows)
    plot_family1_tsr(summary, args.output_dir)
    plot_family2_tsr(summary, args.output_dir)
    write_summary_table(summary, args.output_dir)


if __name__ == "__main__":
    main()
