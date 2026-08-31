"""Render a real FoodReg test-case study for the SPSR-Net paper.

The figure is deliberately a qualitative prediction example rather than a
bar-chart restatement of the ablation table.  It uses the sigmoid score for
the exact, gold IV span emitted by the two saved models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


CASE_SPAN = "与参考食品比较，矿物质含量增加25%以上(含25%)"
FULL_MODEL = "Full SPSR-Net"
ABLATION_MODEL = "w/o SNSA"
THRESHOLD = 0.48


def get_case_scores(path: Path) -> tuple[float, float]:
    """Read the verified scores for one FoodReg test-set annotation."""
    data = json.loads(path.read_text(encoding="utf-8"))
    scores: dict[str, float] = {}
    for model in (FULL_MODEL, ABLATION_MODEL):
        for record in data[model]:
            for gold in record["gold"]:
                if gold["label"] == "IV" and gold["text"] == CASE_SPAN:
                    scores[model] = gold["probability"]
                    break
            if model in scores:
                break
    if set(scores) != {FULL_MODEL, ABLATION_MODEL}:
        raise RuntimeError("The selected FoodReg IV case was not found in the score export.")
    return scores[FULL_MODEL], scores[ABLATION_MODEL]


def rounded_box(ax, x, y, width, height, *, facecolor, edgecolor, linewidth=1.2, radius=0.02):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def add_prediction_card(
    ax,
    x,
    *,
    model: str,
    score: float,
    accepted: bool,
    accent: str,
    background: str,
) -> None:
    """Add one model's actual decision for the same exact gold IV span."""
    rounded_box(ax, x, 0.16, 0.405, 0.305, facecolor=background, edgecolor=accent, linewidth=1.6)
    ax.text(x + 0.025, 0.413, model, transform=ax.transAxes, fontsize=12.5, weight="bold", color="#182532")
    ax.text(
        x + 0.025,
        0.362,
        "Exact gold IV span retained" if accepted else "Exact gold IV span rejected",
        transform=ax.transAxes,
        fontsize=10.6,
        color="#2E3D4B",
    )
    ax.text(x + 0.025, 0.260, f"p(IV) = {score:.3f}", transform=ax.transAxes, fontsize=23, weight="bold", color=accent)
    relation = ">" if accepted else "<"
    ax.text(
        x + 0.025,
        0.210,
        f"{relation} decision threshold  τ = {THRESHOLD:.2f}",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#435466",
    )
    outcome = "Correct extraction of the nutrition claim" if accepted else "False negative for the nutrition claim"
    ax.text(x + 0.025, 0.177, outcome, transform=ax.transAxes, fontsize=9.8, color="#435466")


def render(output_dir: Path, score_path: Path) -> None:
    full_score, ablated_score = get_case_scores(score_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(10.6, 6.5), facecolor="white")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Figure heading.
    ax.text(
        0.05,
        0.945,
        "Food-safety regulation case: extracting a nutrient-content claim",
        transform=ax.transAxes,
        fontsize=17,
        weight="bold",
        color="#17212B",
    )
    ax.text(
        0.05,
        0.902,
        "FoodReg test set · complex indicator value (IV) with a comparison target and percentage constraint",
        transform=ax.transAxes,
        fontsize=10.2,
        color="#5B6B7A",
    )

    # Test clause and its gold annotation.  The English wording is an exact
    # semantic translation of the Chinese FoodReg source clause.
    rounded_box(ax, 0.05, 0.585, 0.90, 0.245, facecolor="#F7F9FB", edgecolor="#CED7E0", linewidth=1.0)
    ax.text(0.075, 0.786, "FoodReg test clause (translated)", transform=ax.transAxes, fontsize=10.5, weight="bold", color="#3B4A5A")
    ax.text(
        0.075,
        0.732,
        "Nutrient condition: increase in minerals (excluding sodium).",
        transform=ax.transAxes,
        fontsize=11.2,
        color="#1E2A36",
    )
    ax.text(0.075, 0.682, "Gold indicator value (IV):", transform=ax.transAxes, fontsize=10.5, weight="bold", color="#9A5A00")
    rounded_box(ax, 0.266, 0.643, 0.650, 0.075, facecolor="#FFF0D5", edgecolor="#E5A647", linewidth=1.1, radius=0.012)
    ax.text(
        0.286,
        0.671,
        "Compared with reference food, mineral content increases by ≥25% (inclusive).",
        transform=ax.transAxes,
        fontsize=10.3,
        weight="bold",
        color="#573A09",
    )
    ax.text(
        0.075,
        0.608,
        "The highlighted span combines a reference object, a nutrient component, and a regulatory threshold.",
        transform=ax.transAxes,
        fontsize=9.7,
        color="#5B6B7A",
    )

    # Route the same gold span to the two model decisions.
    for target_x in (0.275, 0.725):
        arrow = FancyArrowPatch(
            (0.59, 0.584),
            (target_x, 0.485),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.15,
            linestyle=(0, (3, 3)),
            color="#8A98A6",
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(arrow)

    add_prediction_card(
        ax,
        0.05,
        model=FULL_MODEL,
        score=full_score,
        accepted=full_score >= THRESHOLD,
        accent="#0B7A6F",
        background="#EFFAF7",
    )
    add_prediction_card(
        ax,
        0.545,
        model=ABLATION_MODEL,
        score=ablated_score,
        accepted=ablated_score >= THRESHOLD,
        accent="#BD3D4B",
        background="#FFF4F4",
    )

    ax.text(
        0.05,
        0.070,
        "Interpretation: on this real food-safety rule, SNSA changes the exact-span decision from a missed IV to a retained IV.",
        transform=ax.transAxes,
        fontsize=10.1,
        color="#304454",
    )
    ax.text(
        0.05,
        0.032,
        "Scores are sigmoid probabilities for the exact gold span from the saved Full and w/o SNSA checkpoints; this is a case study, not an aggregate metric.",
        transform=ax.transAxes,
        fontsize=8.3,
        color="#6B7B89",
    )

    fig.savefig(output_dir / "Figure2_FoodReg_case_study.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_dir / "Figure2_FoodReg_case_study.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_dir / "Figure2_FoodReg_case_study.svg", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path(__file__).with_name("food_gold_probability_profiles.json"),
    )
    args = parser.parse_args()
    render(args.output_dir, args.scores)


if __name__ == "__main__":
    main()
