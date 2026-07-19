import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Family 1 observation-noise test results.")
    parser.add_argument(
        "--results-dir",
        default=str(ROOT / "results" / "experiment_family_1_obsnoise"),
        help="Directory containing noisy-eval test CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "plots" / "thesis_test_results_obsnoise"),
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
    text = " ".join([str(row.get("exp_name", "")), str(row.get("_source_path", ""))]).lower()
    if "plpg_fine" in text:
        return "PLPG Fine"
    if "plpg" in text:
        return "PLPG"
    if "ppo" in text:
        return "PPO"
    return str(row.get("method", "Unknown"))


def infer_k(row):
    return int(float(row.get("num_rl_agents", 0)))


def to_float(value):
    if value in (None, ""):
        return 0.0
    return float(value)


def build_summary(rows):
    summary = {}
    for row in rows:
        variant = infer_variant(row)
        k_value = infer_k(row)
        summary[(variant, k_value)] = {
            "tsr": to_float(row.get("tsr")),
            "failure_rate": to_float(row.get("failure_rate")),
            "timeout_rate": to_float(row.get("timeout_rate")),
            "mean_reward": to_float(row.get("mean_reward")),
        }
    return summary


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_metric(summary, metric_key, y_label, title, output_path, ylim=None):
    variants = ["PPO", "PLPG", "PLPG Fine"]
    ks = [1, 2, 3]
    plt.figure(figsize=(7, 4.5))
    for variant in variants:
        xs = [k for k in ks if (variant, k) in summary]
        ys = [summary[(variant, k)][metric_key] for k in xs]
        if xs:
            plt.plot(xs, ys, marker="o", linewidth=2, label=variant)
    plt.xticks(ks)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Number of RL Agents (K)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_summary_table(summary, output_dir):
    output_path = Path(output_dir) / "test_summary_obsnoise.tsv"
    with open(output_path, "w", newline="") as handle:
        handle.write("variant\tk\ttsr\tfailure_rate\ttimeout_rate\tmean_reward\n")
        for variant, k_value in sorted(summary.keys(), key=lambda item: (item[0], item[1])):
            metrics = summary[(variant, k_value)]
            handle.write(
                "\t".join(
                    [
                        variant,
                        str(k_value),
                        f"{metrics['tsr']:.4f}",
                        f"{metrics['failure_rate']:.4f}",
                        f"{metrics['timeout_rate']:.4f}",
                        f"{metrics['mean_reward']:.4f}",
                    ]
                )
                + "\n"
            )


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    csv_paths = sorted(Path(args.results_dir).rglob("*_test_metrics.csv"))
    rows = read_rows(csv_paths)
    summary = build_summary(rows)

    plot_metric(
        summary,
        "tsr",
        "TSR",
        "Family 1 Observation-Noise Benchmark",
        Path(args.output_dir) / "family1_obsnoise_tsr.png",
        ylim=(0.0, 1.05),
    )
    plot_metric(
        summary,
        "failure_rate",
        "Failure Rate",
        "Family 1 Observation-Noise Failure Rate",
        Path(args.output_dir) / "family1_obsnoise_failure_rate.png",
        ylim=(0.0, 1.05),
    )
    plot_metric(
        summary,
        "timeout_rate",
        "Timeout Rate",
        "Family 1 Observation-Noise Timeout Rate",
        Path(args.output_dir) / "family1_obsnoise_timeout_rate.png",
        ylim=(0.0, 1.05),
    )
    write_summary_table(summary, args.output_dir)


if __name__ == "__main__":
    main()
