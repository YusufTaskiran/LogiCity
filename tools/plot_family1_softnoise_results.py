import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Family 1 soft-noise test results.")
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "experiment_family_1_softnoise"))
    parser.add_argument("--output-dir", default=str(ROOT / "plots" / "thesis_test_results_softnoise"))
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
    return 0.0 if value in (None, "") else float(value)


def build_summary(rows):
    summary = {}
    for row in rows:
        summary[(infer_variant(row), infer_k(row))] = {
            "tsr": to_float(row.get("tsr")),
            "failure_rate": to_float(row.get("failure_rate")),
            "timeout_rate": to_float(row.get("timeout_rate")),
            "mean_reward": to_float(row.get("mean_reward")),
        }
    return summary


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_metric(summary, metric_key, title, y_label, output_path):
    variants = ["PPO", "PLPG", "PLPG Fine"]
    ks = [1, 2, 3]
    plt.figure(figsize=(7, 4.5))
    for variant in variants:
        xs = [k for k in ks if (variant, k) in summary]
        ys = [summary[(variant, k)][metric_key] for k in xs]
        if xs:
            plt.plot(xs, ys, marker="o", linewidth=2, label=variant)
    plt.xticks(ks)
    plt.ylim(0.0, 1.05 if metric_key != "mean_reward" else max(ys + [1.0]) + 0.5)
    plt.xlabel("Number of RL Agents (K)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_summary(summary, output_dir):
    path = Path(output_dir) / "test_summary_softnoise.tsv"
    with open(path, "w", newline="") as handle:
        handle.write("variant\tk\ttsr\tfailure_rate\ttimeout_rate\tmean_reward\n")
        for variant, k_value in sorted(summary.keys(), key=lambda item: (item[0], item[1])):
            m = summary[(variant, k_value)]
            handle.write(
                f"{variant}\t{k_value}\t{m['tsr']:.4f}\t{m['failure_rate']:.4f}\t{m['timeout_rate']:.4f}\t{m['mean_reward']:.4f}\n"
            )


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    rows = read_rows(sorted(Path(args.results_dir).rglob("*_test_metrics.csv")))
    summary = build_summary(rows)
    plot_metric(summary, "tsr", "Family 1 Soft-Noise TSR", "TSR", Path(args.output_dir) / "family1_softnoise_tsr.png")
    plot_metric(summary, "failure_rate", "Family 1 Soft-Noise Failure Rate", "Failure Rate", Path(args.output_dir) / "family1_softnoise_failure_rate.png")
    plot_metric(summary, "timeout_rate", "Family 1 Soft-Noise Timeout Rate", "Timeout Rate", Path(args.output_dir) / "family1_softnoise_timeout_rate.png")
    write_summary(summary, args.output_dir)


if __name__ == "__main__":
    main()
