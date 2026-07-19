import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def read_tsv(path):
    with open(path, "r", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def build_perfect_series(rows):
    data = {}
    for row in rows:
        if row["difficulty"] != "easy":
            continue
        variant = row["variant"]
        if variant not in {"PPO", "PLPG"}:
            continue
        data[(variant, int(row["k"]))] = float(row["tsr"])
    return data


def build_uncertain_series(rows):
    data = {}
    for row in rows:
        variant = row["variant"]
        if variant not in {"PPO", "PLPG"}:
            continue
        data[(variant, int(row["k"]))] = float(row["tsr"])
    return data


def plot_tsr(data, title, output_path):
    colors = {"PPO": "#1f77b4", "PLPG": "#d62728"}
    ks = [1, 2, 3]
    plt.figure(figsize=(6.8, 4.2))
    for variant in ["PPO", "PLPG"]:
        xs = [k for k in ks if (variant, k) in data]
        ys = [data[(variant, k)] for k in xs]
        plt.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=colors[variant],
            label=variant,
        )
    plt.xticks(ks)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Number of RL Agents (K)")
    plt.ylabel("TSR")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()


def main():
    perfect_rows = read_tsv(ROOT / "plots" / "thesis_test_results" / "test_summary.tsv")
    uncertain_rows = read_tsv(ROOT / "plots" / "thesis_test_results_obsnoise" / "test_summary_obsnoise.tsv")

    figures_dir = ROOT / "thesis" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_tsr(
        build_perfect_series(perfect_rows),
        "Experiment 1: Perfect Symbolic Observations",
        figures_dir / "exp1_perfect_tsr.png",
    )
    plot_tsr(
        build_uncertain_series(uncertain_rows),
        "Experiment 1: Uncertain Symbolic Observations",
        figures_dir / "exp1_uncertain_tsr.png",
        )


if __name__ == "__main__":
    main()
