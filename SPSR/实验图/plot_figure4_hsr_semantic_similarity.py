"""Render an HSR semantic-similarity heatmap for one FoodReg rule.

Values are cosine similarities between real span representations immediately
before the classifier.  The comparison is Full SPSR-Net versus its w/o HSR
checkpoint on the same FoodReg test clause.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from plot_figure3_food_semantic_similarity import (  # noqa: E402
    FULL_CHECKPOINT,
    MODEL_NAME,
    THRESHOLD,
    build_args,
    colour,
    labels,
    load_font,
)

PROJECT = Path("/root/shared-nvme/projects/SPSR-Net")
for entry in (PROJECT, PROJECT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
from eval_thresholds import build_model, load_data, move_to_device  # noqa: E402


NO_HSR_CHECKPOINT = (
    PROJECT / "_saved_models/2026-07-05-22_12_33_633670/"
    "model-epoch_8-batch_24008-f#f#test_95.87/fastnlp_model.pkl.tar"
)
CASE_TEXT = (
    "增加矿物质(不包括钠)的指标值为与参考食品比较，矿物质含量增加25%以上(含25%)，"
    "检验方法为参考食品的数据来源：1.同一企业同类或同一属类食品的营养成分含量或2.《中国食物成分表》中同类食品营养成分含量。"
)
UNITS = (
    ("Mineral increase (IN)", "增加矿物质(不包括钠)"),
    ("Comparison", "与参考食品比较"),
    ("Reference food", "参考食品"),
    ("Mineral", "矿物质"),
    ("Increase", "含量增加"),
    ("Threshold ≥25%", "25%以上"),
    ("Inclusive condition", "含25%"),
    ("Gold IV span", "与参考食品比较，矿物质含量增加25%以上(含25%)"),
)
ANCHORS = (0, 7)


def find_case(dataloader):
    for batch in dataloader:
        for row, words in enumerate(batch["raw_words"]):
            if "".join(words) == CASE_TEXT:
                return batch, row
    raise RuntimeError("Selected FoodReg test clause was not found.")


def spans_for_case(tokens: list[str]) -> list[tuple[int, int]]:
    text = "".join(tokens)
    spans = []
    for _, phrase in UNITS:
        start = text.find(phrase)
        if start < 0:
            raise RuntimeError(f"Cannot locate phrase: {phrase}")
        spans.append((start, start + len(phrase) - 1))
    return spans


@torch.no_grad()
def forward(model, batch, device):
    captured = []

    def hook(_module, inputs, _output):
        captured.append(inputs[0].detach().cpu())

    handle = model.score_head.register_forward_hook(hook)
    try:
        device_batch = move_to_device(batch, device)
        logits = model(
            input_ids=device_batch["input_ids"],
            bpe_len=device_batch["bpe_len"],
            indexes=device_batch["indexes"],
            matrix=device_batch["matrix"],
            raw_words=device_batch["raw_words"],
        )["scores"].detach().cpu()
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("Could not capture pre-classifier span representations.")
    return captured[0], logits


def evaluate(checkpoint: Path, use_hsr: bool, batch, row: int, matrix_segs, device, spans, iv_index: int):
    args = build_args(checkpoint, True)
    args.use_hsr = int(use_hsr)
    model = build_model(MODEL_NAME, matrix_segs, args).to(device).eval()
    features, logits = forward(model, batch, device)
    vectors = torch.stack([features[row, start, end] for start, end in spans])
    vectors = F.normalize(vectors.float(), dim=-1, eps=1e-8)
    similarity = vectors @ vectors.T
    matrix = [[float(value) for value in current] for current in similarity[:, list(ANCHORS)]]
    probability = logits.sigmoid()
    probability = (probability + probability.transpose(1, 2)) / 2
    iv_start, iv_end = spans[ANCHORS[1]]
    iv_probability = float(probability[row, iv_start, iv_end, iv_index])
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return matrix, iv_probability


def draw_panel(draw, matrix, panel_caption, x0: int, y0: int):
    """Draw one compact panel in the supplied-paper heatmap style."""
    caption_font = load_font(17)
    row_font = load_font(14)
    column_font = load_font(15, True)
    cell_font = load_font(14)
    # Short, wide cells reproduce the compact paper-style heatmap proportions.
    cell_w, cell_h = 190, 30
    label_width = 240
    grid_x, grid_y = x0 + label_width, y0
    columns = ("Nutrient\ncondition", "Complete\nIV")
    for column, label in enumerate(columns):
        box = draw.multiline_textbbox((0, 0), label, font=column_font, spacing=0)
        width = box[2] - box[0]
        draw.multiline_text(
            (grid_x + column * cell_w + (cell_w - width) / 2, grid_y - 38),
            label,
            fill=(38, 45, 52),
            font=column_font,
            align="center",
            spacing=0,
        )
    for row, ((label, _), values) in enumerate(zip(UNITS, matrix)):
        label_box = draw.textbbox((0, 0), label, font=row_font)
        draw.text((grid_x - 14 - (label_box[2] - label_box[0]), grid_y + row * cell_h + 8), label, fill=(30, 37, 43), font=row_font)
        for column, value in enumerate(values):
            x = grid_x + column * cell_w
            y = grid_y + row * cell_h
            draw.rectangle(
                (x, y, x + cell_w, y + cell_h),
                fill=colour(value, minimum=0.40, maximum=1.00),
                outline=(247, 247, 247),
                width=2,
            )
            number = "1.00" if value >= 0.995 else f"{value:.2f}"
            number_box = draw.textbbox((0, 0), number, font=cell_font)
            foreground = (255, 255, 255) if value >= 0.71 or value <= 0.34 else (28, 34, 39)
            draw.text(
                (x + (cell_w - (number_box[2] - number_box[0])) / 2, y + (cell_h - (number_box[3] - number_box[1])) / 2 - 1),
                number,
                fill=foreground,
                font=cell_font,
            )
    caption_box = draw.textbbox((0, 0), panel_caption, font=caption_font)
    caption_x = grid_x + cell_w - (caption_box[2] - caption_box[0]) / 2
    draw.text((caption_x, grid_y + len(UNITS) * cell_h + 17), panel_caption, fill=(30, 37, 43), font=caption_font)
    return grid_x, grid_y + len(UNITS) * cell_h


def draw_colourbar(draw, x: int, y: int, height: int):
    font = load_font(11)
    for pixel in range(height):
        value = 1.0 - pixel / max(1, height - 1) * 0.6
        draw.line((x, y + pixel, x + 16, y + pixel), fill=colour(value, minimum=0.40, maximum=1.00), width=1)
    draw.rectangle((x, y, x + 16, y + height), outline=(55, 55, 55), width=1)
    for value in (1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40):
        position = y + (1 - value) / 0.6 * height
        draw.line((x + 16, position, x + 21, position), fill=(55, 55, 55), width=1)
        draw.text((x + 27, position - 7), f"{value:.1f}", fill=(53, 61, 68), font=font)


def render(output_dir: Path, no_hsr, full, no_hsr_probability: float, full_probability: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (2100, 390), "white")
    draw = ImageDraw.Draw(image)
    draw_panel(draw, no_hsr, "(a) SPSR-Net w/o HSR", 95, 62)
    draw_colourbar(draw, 730, 62, 240)
    draw_panel(draw, full, "(b) Full SPSR-Net", 1115, 62)
    draw_colourbar(draw, 1750, 62, 240)
    image.save(output_dir / "Figure4_FoodReg_HSR_semantic_similarity.png", dpi=(300, 300))
    image.save(output_dir / "Figure4_FoodReg_HSR_semantic_similarity.pdf", resolution=300.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, matrix_segs = load_data(MODEL_NAME, "food", batch_size=4, num_workers=1)
    batch, row = find_case(loaders["test"])
    tokens = list(batch["raw_words"][row])[: int(batch["word_len"][row])]
    spans = spans_for_case(tokens)
    iv_index = labels().index("IV")
    no_hsr, no_hsr_probability = evaluate(NO_HSR_CHECKPOINT, False, batch, row, matrix_segs, device, spans, iv_index)
    full, full_probability = evaluate(FULL_CHECKPOINT, True, batch, row, matrix_segs, device, spans, iv_index)
    render(args.output_dir, no_hsr, full, no_hsr_probability, full_probability)
    report = {
        "case_text": CASE_TEXT,
        "units": [{"name": name, "text": text, "span": list(span)} for (name, text), span in zip(UNITS, spans)],
        "anchor_units": [UNITS[index][0] for index in ANCHORS],
        "matrices": {"w/o HSR": no_hsr, "Full SPSR-Net": full},
        "exact_gold_iv_probability": {"w/o HSR": no_hsr_probability, "Full SPSR-Net": full_probability},
        "threshold": THRESHOLD,
        "feature_source": "Input to score_head (span representation immediately before final classification).",
        "checkpoints": {"w/o HSR": str(NO_HSR_CHECKPOINT), "Full SPSR-Net": str(FULL_CHECKPOINT)},
    }
    report_path = args.output_dir / "Figure4_FoodReg_HSR_semantic_similarity_data.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figure": str(args.output_dir / "Figure4_FoodReg_HSR_semantic_similarity.png"), "report": str(report_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
