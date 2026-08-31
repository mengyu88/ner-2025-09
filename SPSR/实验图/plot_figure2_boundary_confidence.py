#!/usr/bin/env python3
"""Plot a FoodReg case study from exported, real span probabilities.

Unlike a semantic-similarity heatmap, this figure visualizes the model's own
class probability for spans with endpoints close to the gold boundary.  The
measure directly corresponds to the strict span-classification decision used
in SPSR-Net evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "food_case_span_scores.json"
TARGET_TEXT = "与参考食品比较，矿物质含量增加25%以上(含25%)"
TARGET_LABEL = "IV"
THRESHOLD = 0.48
OFFSETS = np.arange(-2, 3)


def read_target() -> tuple[list[dict], int, int, int]:
    records = json.loads(SOURCE.read_text(encoding="utf-8"))
    first = records[0]
    label_id = first["labels"].index(TARGET_LABEL)
    target = next(item for item in first["gold"] if item["text"] == TARGET_TEXT and item["label"] == TARGET_LABEL)
    return records, int(target["start"]), int(target["end"]), label_id


def endpoint_window(record: dict, start: int, end: int, label_id: int) -> np.ndarray:
    probabilities = np.asarray(record["probabilities"], dtype=float)
    values = np.full((len(OFFSETS), len(OFFSETS)), np.nan)
    for row, start_offset in enumerate(OFFSETS):
        for col, end_offset in enumerate(OFFSETS):
            left, right = start + start_offset, end + end_offset
            if 0 <= left <= right < probabilities.shape[0]:
                values[row, col] = probabilities[left, right, label_id]
    return values


def main() -> None:
    records, start, end, label_id = read_target()
    windows = [endpoint_window(record, start, end, label_id) for record in records]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titleweight": "bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.8), constrained_layout=True)
    figure_title = "Boundary Confidence for a Long Food-Safety Indicator Value"
    fig.suptitle(figure_title, fontsize=15, fontweight="bold", y=1.035)

    image = None
    for position, (axis, record, values) in enumerate(zip(axes, records, windows)):
        image = axis.imshow(values, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
        axis.set_title(record["model"], fontsize=12, pad=8)
        axis.set_xticks(range(len(OFFSETS)), [f"e{offset:+d}" for offset in OFFSETS])
        axis.set_yticks(range(len(OFFSETS)), [f"s{offset:+d}" for offset in OFFSETS])
        axis.set_xlabel("End-boundary offset from gold")
        if position == 0:
            axis.set_ylabel("Start-boundary offset from gold")

        for row in range(len(OFFSETS)):
            for col in range(len(OFFSETS)):
                value = values[row, col]
                if np.isnan(value):
                    label = "–"
                elif value < 0.005:
                    label = "0.00"
                else:
                    label = f"{value:.2f}"
                color = "white" if value >= 0.56 else "#29333B"
                axis.text(col, row, label, ha="center", va="center", fontsize=9.5, color=color)

        # The central cell is the exact gold start and end boundary.
        axis.add_patch(Rectangle((1.5, 1.5), 1, 1, fill=False, lw=2.4, edgecolor="#183A4A"))
        gold_score = values[2, 2]
        status = "retained" if gold_score >= THRESHOLD else "rejected"
        axis.text(
            0.5,
            -0.28,
            f"Gold IV span: p = {gold_score:.2f}  →  {status} at τ = {THRESHOLD:.2f}",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontweight="bold",
            color="#0B6E69" if gold_score >= THRESHOLD else "#B23A48",
        )

    colorbar = fig.colorbar(image, ax=axes, shrink=0.89, pad=0.02)
    colorbar.set_label("Predicted probability for indicator value (IV)", rotation=90)
    fig.text(
        0.5,
        -0.065,
        "Each cell is one candidate span. s/e indicate start/end shifts relative to the gold boundary; "
        "the outlined centre cell is the exact gold span.",
        ha="center",
        va="top",
        fontsize=9,
        color="#52616B",
    )

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(HERE / f"Figure2_FoodReg_boundary_confidence.{suffix}", dpi=360, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
