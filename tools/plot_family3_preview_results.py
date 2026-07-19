import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ["PPO", "PLPG", "PLPG Fine"]
MODEL_LABELS = {
    "ppo": "PPO",
    "plpg": "PLPG",
    "plpg_fine": "PLPG Fine",
    "PPO": "PPO",
    "PLPGPPO": "PLPG",
}
K_ORDER = [5, 10, 15, 20]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot family-3 preview rollout TSR results.")
    parser.add_argument(
        "--results-dir",
        default=str(ROOT / "results" / "experiment_family_3_preview300"),
        help="Directory containing family-3 rollout metric CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "plots" / "family3_preview300"),
        help="Directory where plots and merged summaries will be written.",
    )
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_model(row):
    exp_name = str(row.get("exp_name", "")).lower()
    if "plpg_fine" in exp_name:
        return "PLPG Fine"
    if "plpg" in exp_name:
        return "PLPG"
    if "ppo" in exp_name:
        return "PPO"
    return MODEL_LABELS.get(str(row.get("method", "")), str(row.get("method", "Unknown")))


def to_float(value):
    if value in (None, ""):
        return 0.0
    return float(value)


def to_int(value):
    if value in (None, ""):
        return 0
    return int(float(value))


def load_rows(results_dir):
    rows = []
    for csv_path in sorted(Path(results_dir).rglob("*_rollout_metrics.csv")):
        with open(csv_path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["_source_path"] = str(csv_path)
                row["model"] = normalize_model(row)
                row["difficulty"] = str(row.get("difficulty", "")).lower()
                row["k"] = to_int(row.get("k"))
                row["micro_rollout_tsr"] = to_float(row.get("micro_rollout_tsr"))
                row["macro_rollout_tsr"] = to_float(row.get("macro_rollout_tsr"))
                rows.append(row)
    return rows


def build_summary(rows):
    summary = {}
    for row in rows:
        key = (row["difficulty"], row["model"], row["k"])
        summary[key] = {
            "exp_name": row.get("exp_name", ""),
            "micro_rollout_tsr": row["micro_rollout_tsr"],
            "macro_rollout_tsr": row["macro_rollout_tsr"],
            "resolved_goal_attempts": to_int(row.get("resolved_goal_attempts")),
            "successful_goal_attempts": to_int(row.get("successful_goal_attempts")),
            "failed_goal_attempts": to_int(row.get("failed_goal_attempts")),
            "steps_to_resolution": to_int(row.get("steps_to_resolution")),
            "source_path": row["_source_path"],
        }
    return summary


def write_summary_table(summary, output_dir):
    output_path = Path(output_dir) / "family3_preview300_summary.tsv"
    with open(output_path, "w", newline="") as handle:
        handle.write(
            "difficulty\tmodel\tk\tmicro_rollout_tsr\tmacro_rollout_tsr\tresolved_goal_attempts\tsuccessful_goal_attempts\tfailed_goal_attempts\tsteps_to_resolution\texp_name\tsource_path\n"
        )
        for difficulty, model, k_value in sorted(summary.keys(), key=lambda item: (item[0], item[1], item[2])):
            metrics = summary[(difficulty, model, k_value)]
            handle.write(
                "\t".join(
                    [
                        difficulty,
                        model,
                        str(k_value),
                        f"{metrics['micro_rollout_tsr']:.4f}",
                        f"{metrics['macro_rollout_tsr']:.4f}",
                        str(metrics["resolved_goal_attempts"]),
                        str(metrics["successful_goal_attempts"]),
                        str(metrics["failed_goal_attempts"]),
                        str(metrics["steps_to_resolution"]),
                        metrics["exp_name"],
                        metrics["source_path"],
                    ]
                )
                + "\n"
            )


def available_difficulties(summary):
    return [difficulty for difficulty in ["easy", "medium", "hard"] if any(key[0] == difficulty for key in summary.keys())]


def plot_models_per_k(summary, difficulty, metric_key, metric_label, output_path):
    ks = [k for k in K_ORDER if any((difficulty, model, k) in summary for model in MODEL_ORDER)]
    if not ks:
        return
    x_positions = list(range(len(ks)))
    width = 0.24
    offsets = {
        "PPO": -width,
        "PLPG": 0.0,
        "PLPG Fine": width,
    }

    plt.figure(figsize=(8, 4.8))
    for model in MODEL_ORDER:
        xs = []
        ys = []
        for idx, k_value in enumerate(ks):
            if (difficulty, model, k_value) in summary:
                xs.append(x_positions[idx] + offsets[model])
                ys.append(summary[(difficulty, model, k_value)][metric_key])
        if xs:
            plt.bar(xs, ys, width=width, label=model)

    plt.xticks(x_positions, [str(k) for k in ks])
    plt.ylim(0.0, 1.05)
    plt.xlabel("Number of RL Agents (K)")
    plt.ylabel(metric_label)
    plt.title("{}: {} by K (Models Compared per K)".format(difficulty.capitalize(), metric_label))
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_k_per_model(summary, difficulty, metric_key, metric_label, output_path):
    models = [model for model in MODEL_ORDER if any((difficulty, model, k) in summary for k in K_ORDER)]
    if not models:
        return
    x_positions = list(range(len(models)))
    width = 0.18
    offsets = {
        5: -1.5 * width,
        10: -0.5 * width,
        15: 0.5 * width,
        20: 1.5 * width,
    }

    plt.figure(figsize=(8, 4.8))
    for k_value in K_ORDER:
        xs = []
        ys = []
        for idx, model in enumerate(models):
            if (difficulty, model, k_value) in summary:
                xs.append(x_positions[idx] + offsets[k_value])
                ys.append(summary[(difficulty, model, k_value)][metric_key])
        if xs:
            plt.bar(xs, ys, width=width, label="K={}".format(k_value))

    plt.xticks(x_positions, models)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Model")
    plt.ylabel(metric_label)
    plt.title("{}: {} by Model (K Values Compared per Model)".format(difficulty.capitalize(), metric_label))
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    rows = load_rows(args.results_dir)
    summary = build_summary(rows)
    write_summary_table(summary, args.output_dir)

    for difficulty in available_difficulties(summary):
        plot_models_per_k(
            summary,
            difficulty,
            "micro_rollout_tsr",
            "Micro TSR",
            Path(args.output_dir) / "family3_{}_micro_tsr_models_per_k.png".format(difficulty),
        )
        plot_k_per_model(
            summary,
            difficulty,
            "micro_rollout_tsr",
            "Micro TSR",
            Path(args.output_dir) / "family3_{}_micro_tsr_k_per_model.png".format(difficulty),
        )
        plot_models_per_k(
            summary,
            difficulty,
            "macro_rollout_tsr",
            "Macro TSR",
            Path(args.output_dir) / "family3_{}_macro_tsr_models_per_k.png".format(difficulty),
        )
        plot_k_per_model(
            summary,
            difficulty,
            "macro_rollout_tsr",
            "Macro TSR",
            Path(args.output_dir) / "family3_{}_macro_tsr_k_per_model.png".format(difficulty),
        )


if __name__ == "__main__":
    main()
