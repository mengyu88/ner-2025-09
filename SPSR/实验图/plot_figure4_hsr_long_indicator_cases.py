"""Plot real long-IV confidence paths for the HSR FoodReg ablation.

This is a qualitative, four-case analysis.  Every point is the probability of
the exact annotated ``IV`` span, computed with the same sigmoid-then-symmetric
rule as the project evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path("/root/shared-nvme/projects/SPSR-Net")
for entry in (PROJECT, PROJECT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from eval_thresholds import build_model, load_data, move_to_device  # noqa: E402


MODEL_NAME = "/root/.cache/modelscope/hub/AI-ModelScope/bert-base-chinese"
FULL_CHECKPOINT = (
    PROJECT / "_saved_models/2026-07-03-00_07_48_429272/"
    "model-epoch_15-batch_45015-f#f#test_97.98/fastnlp_model.pkl.tar"
)
NO_HSR_CHECKPOINT = (
    PROJECT / "_saved_models/2026-07-05-22_12_33_633670/"
    "model-epoch_8-batch_24008-f#f#test_95.87/fastnlp_model.pkl.tar"
)
THRESHOLD = 0.48

CASES = (
    {
        "id": "Mineral increase",
        "rule": "Mineral content increases by ≥25% versus a reference food",
        "text": "增加矿物质(不包括钠)的指标值为与参考食品比较，矿物质含量增加25%以上(含25%)，检验方法为参考食品的数据来源：1.同一企业同类或同一属类食品的营养成分含量或2.《中国食物成分表》中同类食品营养成分含量。",
    },
    {
        "id": "Energy reduction",
        "rule": "Energy value decreases by ≥25% versus a reference food",
        "text": "减少能量的指标值为与参考食品比较，能量值减少25%以上(含25%)，检验方法为参考食品的数据来源：1.同一企业同类或同一属类食品的营养成分含量或2.《中国食物成分表》中同类食品营养成分含量。",
    },
    {
        "id": "Dietary-fibre claim",
        "rule": "Dietary fibre changes by ≥25% versus a reference food",
        "text": "增加或减少膳食纤维的指标值为与参考食品比较，膳食纤维含量增加或减少25%以上，检验方法为参考食品(基准食品)应为消费者熟知、容易理解的同类或同一属类食品。",
    },
    {
        "id": "Sugar reduction",
        "rule": "Sugar content decreases by ≥25% versus a reference food",
        "text": "减少糖的指标值为与参考食品比较，糖含量减少25%以上，检验方法为参考食品(基准食品)应为消费者熟知、容易理解的同类或同一属类食品。",
    },
)


def build_args(checkpoint: Path, use_hsr: bool) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint=str(checkpoint),
        cnn_dim=400,
        biaffine_size=200,
        n_head=4,
        cnn_depth=1,
        n_layer=2,
        logit_drop=0.15,
        size_embed_dim=25,
        kernel_size=3,
        separateness_rate=0.05,
        theta=1.0,
        sad_topk=2,
        sad_attn_dim=None,
        use_sad=1,
        use_hsr=int(use_hsr),
        sad_use_rel_bias=1,
        sad_gate=1,
        head_type="linear",
        use_length_bias=False,
        length_bias_bins=6,
    )


def foodreg_labels() -> list[str]:
    names: set[str] = set()
    path = PROJECT / "preprocess/outputs/food/train.jsonlines"
    for line in path.read_text(encoding="utf-8").splitlines():
        names.update(mention["entity_type"] for mention in json.loads(line)["entity_mentions"])
    return sorted(names)


def collect_case_batches(dataloader):
    wanted = {case["text"]: case for case in CASES}
    found = {}
    for batch in dataloader:
        for row, words in enumerate(batch["raw_words"]):
            text = "".join(words)
            if text in wanted:
                found[text] = (batch, row)
        if len(found) == len(wanted):
            break
    missing = set(wanted) - set(found)
    if missing:
        raise RuntimeError(f"FoodReg examples not found: {len(missing)}")
    return [(case, *found[case["text"]]) for case in CASES]


@torch.no_grad()
def evaluate_cases(model, case_batches, device: torch.device, iv_id: int):
    scores_by_case = {}
    spans_by_case = {}
    for case, batch, row in case_batches:
        device_batch = move_to_device(batch, device)
        logits = model(
            input_ids=device_batch["input_ids"],
            bpe_len=device_batch["bpe_len"],
            indexes=device_batch["indexes"],
            matrix=device_batch["matrix"],
            raw_words=device_batch["raw_words"],
        )["scores"].detach().cpu()
        probabilities = logits.sigmoid()
        probabilities = (probabilities + probabilities.transpose(1, 2)) / 2
        iv_spans = [(int(start), int(end)) for start, end, label in batch["ent_target"][row] if int(label) == iv_id]
        if len(iv_spans) != 1:
            raise RuntimeError(f"Expected one IV span in {case['id']}, found {len(iv_spans)}")
        start, end = iv_spans[0]
        scores_by_case[case["id"]] = float(probabilities[row, start, end, iv_id])
        spans_by_case[case["id"]] = {
            "start": start,
            "end": end,
            "text": "".join(batch["raw_words"][row][start:end + 1]),
        }
    return scores_by_case, spans_by_case


def load_font(size: int, bold: bool = False):
    paths = (
        ["/base/mambaforge/fonts/Ubuntu-B.ttf", "/base/mambaforge/fonts/DejaVuSans.ttf"]
        if bold
        else ["/base/mambaforge/fonts/Ubuntu-R.ttf", "/base/mambaforge/fonts/DejaVuSans.ttf"]
    )
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_marker(draw, x: float, y: float, colour: tuple[int, int, int], label: str, *, fill: bool):
    radius = 10
    if fill:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colour, outline="white", width=2)
    else:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="white", outline=colour, width=4)
    label_font = load_font(17, bold=True)
    box = draw.textbbox((0, 0), label, font=label_font)
    draw.text((x - (box[2] - box[0]) / 2, y - 42), label, fill=colour, font=label_font)


def render(output_dir: Path, no_hsr, full):
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1840, 1080), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(35, bold=True)
    subtitle_font = load_font(20)
    row_font = load_font(21, bold=True)
    rule_font = load_font(17)
    axis_font = load_font(17)
    small_font = load_font(16)
    note_font = load_font(18)
    no_hsr_colour = (191, 65, 75)
    full_colour = (8, 122, 111)
    axis_colour = (104, 119, 134)
    x_left, x_right = 690, 1670
    minimum, maximum = 0.40, 1.00
    y_values = (365, 520, 675, 830)

    def x_position(score: float) -> float:
        return x_left + (score - minimum) / (maximum - minimum) * (x_right - x_left)

    draw.text((55, 30), "HSR strengthens confidence for long food-safety indicator values", fill=(22, 30, 37), font=title_font)
    draw.text(
        (55, 83),
        "Exact gold IV probabilities for four representative FoodReg test rules; each value uses the evaluator's symmetric decision rule.",
        fill=(89, 104, 118),
        font=subtitle_font,
    )
    draw.rounded_rectangle((55, 127, 1785, 202), radius=13, fill=(248, 249, 251), outline=(208, 216, 224), width=2)
    draw.text(
        (79, 147),
        "Each target span jointly encodes a comparison object, a nutrient or energy component, a direction of change, and a numerical threshold.",
        fill=(43, 56, 67),
        font=note_font,
    )
    draw.text(
        (79, 177),
        "Thus this case analysis complements the ablation table: it illustrates confidence on concrete regulatory claims rather than another aggregate score.",
        fill=(88, 102, 114),
        font=small_font,
    )

    # Axis and threshold.
    for tick in (0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00):
        x = x_position(tick)
        draw.line((x, 295, x, 872), fill=(235, 239, 243), width=2)
        label = f"{tick:.2f}"
        box = draw.textbbox((0, 0), label, font=axis_font)
        draw.text((x - (box[2] - box[0]) / 2, 888), label, fill=axis_colour, font=axis_font)
    threshold_x = x_position(THRESHOLD)
    draw.line((threshold_x, 273, threshold_x, 872), fill=(132, 145, 157), width=3)
    draw.text((threshold_x + 8, 249), "decision threshold τ = 0.48", fill=(92, 105, 117), font=small_font)
    draw.line((x_left, 872, x_right, 872), fill=axis_colour, width=2)
    axis_label = "Probability assigned to the exact gold IV span"
    box = draw.textbbox((0, 0), axis_label, font=axis_font)
    draw.text(((x_left + x_right - (box[2] - box[0])) / 2, 930), axis_label, fill=(53, 67, 78), font=axis_font)

    # Four paired paths.
    for case, y in zip(CASES, y_values):
        no_hsr_score = no_hsr[case["id"]]
        full_score = full[case["id"]]
        no_x, full_x = x_position(no_hsr_score), x_position(full_score)
        draw.line((x_left, y, x_right, y), fill=(225, 230, 235), width=2)
        draw.text((55, y - 44), case["id"], fill=(29, 40, 50), font=row_font)
        draw.text((55, y - 12), case["rule"], fill=(82, 96, 108), font=rule_font)
        draw.line((no_x, y, full_x - 15, y), fill=(111, 138, 145), width=5)
        draw.polygon(((full_x - 15, y - 8), (full_x, y), (full_x - 15, y + 8)), fill=(111, 138, 145))
        draw_marker(draw, no_x, y, no_hsr_colour, f"{no_hsr_score:.3f}", fill=False)
        draw_marker(draw, full_x, y, full_colour, f"{full_score:.3f}", fill=True)
        delta = full_score - no_hsr_score
        draw.text(((no_x + full_x) / 2 - 30, y + 21), f"+{delta:.3f}", fill=(54, 111, 104), font=small_font)

    # Legend and conclusion.
    draw_marker(draw, 85, 991, no_hsr_colour, "", fill=False)
    draw.text((108, 979), "w/o HSR", fill=(50, 63, 73), font=axis_font)
    draw_marker(draw, 245, 991, full_colour, "", fill=True)
    draw.text((268, 979), "Full SPSR-Net", fill=(50, 63, 73), font=axis_font)
    draw.text(
        (510, 978),
        "Across these rules, HSR raises the confidence of the complete regulatory value span while keeping the same decision threshold.",
        fill=(50, 63, 73),
        font=axis_font,
    )
    draw.text(
        (55, 1032),
        "Qualitative case study only. The overall effect is reported separately by the FoodReg ablation results.",
        fill=(104, 116, 127),
        font=small_font,
    )
    image.save(output_dir / "Figure4_FoodReg_HSR_long_IV_confidence.png", dpi=(300, 300))
    image.save(output_dir / "Figure4_FoodReg_HSR_long_IV_confidence.pdf", resolution=300.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, matrix_segs = load_data(MODEL_NAME, "food", batch_size=4, num_workers=1)
    case_batches = collect_case_batches(loaders["test"])
    iv_id = foodreg_labels().index("IV")

    no_hsr_model = build_model(MODEL_NAME, matrix_segs, build_args(NO_HSR_CHECKPOINT, False)).to(device).eval()
    no_hsr, spans = evaluate_cases(no_hsr_model, case_batches, device, iv_id)
    del no_hsr_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    full_model = build_model(MODEL_NAME, matrix_segs, build_args(FULL_CHECKPOINT, True)).to(device).eval()
    full, _ = evaluate_cases(full_model, case_batches, device, iv_id)
    del full_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    render(args.output_dir, no_hsr, full)
    report = {
        "threshold": THRESHOLD,
        "scores": {"w/o HSR": no_hsr, "Full SPSR-Net": full},
        "gold_spans": spans,
        "cases": CASES,
        "probability_rule": "sigmoid(logits), then average both span directions",
        "checkpoints": {"w/o HSR": str(NO_HSR_CHECKPOINT), "Full SPSR-Net": str(FULL_CHECKPOINT)},
    }
    report_path = args.output_dir / "Figure4_FoodReg_HSR_long_IV_confidence_data.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figure": str(args.output_dir / "Figure4_FoodReg_HSR_long_IV_confidence.png"), "report": str(report_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
