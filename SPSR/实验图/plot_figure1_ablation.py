#!/usr/bin/env python3
"""Render Figure 1 from the FoodReg ablation results reported in the paper.

The chart deliberately encodes the F1 loss relative to the complete SPSR-Net
instead of just repeating the ablation table as a bar chart.  It supports the
claim that each proposed component contributes to strict entity recognition on
the FoodReg test set.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


ROOT = Path("/root/shared-nvme")
SOURCE = ROOT / "projects/SPSR-Net/outputs/food/ablation_20260706/difinet_food_ablation_paper_style_best_test.csv"
OUT = Path(__file__).resolve().parent

ORDER = [
    ("DiFiNet full model", "Full SPSR-Net", "#0B6E69"),
    ("w/o SNSA", "w/o SNSA", "#457B9D"),
    ("w/o HSR", "w/o HSR", "#E76F51"),
    ("w/o PGD-AT", "w/o PGD-AT", "#7B2CBF"),
]


def load_results() -> dict[str, float]:
    with SOURCE.open(encoding="utf-8", newline="") as f:
        return {row["settings"]: float(row["f1"]) for row in csv.DictReader(f)}


def main() -> None:
    values = load_results()
    full_f1 = values["DiFiNet full model"]
    labels = [item[1] for item in ORDER]
    f1_scores = [values[item[0]] for item in ORDER]
    drops = [full_f1 - score for score in f1_scores]
    colors = [item[2] for item in ORDER]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(8.2, 4.65), constrained_layout=True)
    bars = ax.bar(labels, drops, width=0.62, color=colors, edgecolor="white", linewidth=1.2)

    ax.set_ylim(0, 2.55)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.grid(axis="y", color="#D9E1E8", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#7A8691")
    ax.set_ylabel("F1 decrease vs. full model (percentage points)")
    ax.set_title("Component Contribution on FoodReg Test Set", pad=12)

    for bar, f1, drop in zip(bars, f1_scores, drops):
        x = bar.get_x() + bar.get_width() / 2
        if drop == 0:
            ax.text(x, 0.11, f"F1 = {f1:.2f}", ha="center", va="bottom", color="#0B6E69", fontweight="bold")
        else:
            ax.text(x, drop + 0.07, f"−{drop:.2f} pp", ha="center", va="bottom", fontweight="bold", color="#25313C")
            ax.text(x, max(0.08, drop - 0.14), f"F1 = {f1:.2f}", ha="center", va="top", color="white", fontsize=9, fontweight="bold")

    ax.text(
        0.995,
        -0.24,
        "Strict entity-level F1; exact span and type match.  Higher is better.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#52616B",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"Figure1_FoodReg_ablation.{suffix}", dpi=360, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
