"""Render a FoodReg semantic-similarity case study from real model features.

The diagram follows the compact two-column heatmap form commonly used for
case-level representation analysis.  Unlike a classification-score heatmap,
each value is the cosine similarity between two *span representations* fed to
the final classifier.  All values are obtained by an inference pass through
the saved Full SPSR-Net and w/o SNSA checkpoints.
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
import torch.nn.functional as F
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
NO_SNSA_CHECKPOINT = (
    PROJECT / "_saved_models/2026-07-04-20_43_24_622169/"
    "model-epoch_6-batch_18006-f#f#test_97.13/fastnlp_model.pkl.tar"
)
THRESHOLD = 0.48

# The example was selected from the FoodReg test set because the exact IV is
# retained by the full model (p=0.979) but missed without SNSA (p=0.194).
CASE_TEXT = (
    "减少糖的指标值为与参考食品比较，糖含量减少25%以上，"
    "检验方法为参考食品(基准食品)应为消费者熟知、容易理解的同类或同一属类食品。"
)
UNITS = (
    ("Reduce sugar (IN)", "减少糖"),
    ("Comparison", "与参考食品比较"),
    ("Reference food", "参考食品"),
    ("Sugar", "糖"),
    ("Reduction", "含量减少"),
    ("Threshold ≥25%", "25%以上"),
    ("Gold IV span", "与参考食品比较，糖含量减少25%以上"),
)
ANCHOR_INDICES = (0, 6)


def build_args(checkpoint: Path, use_snsa: bool) -> SimpleNamespace:
    """Match the configuration used for the paired FoodReg checkpoints."""
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
        use_sad=int(use_snsa),
        use_hsr=1,
        sad_use_rel_bias=1,
        sad_gate=1,
        head_type="linear",
        use_length_bias=False,
        length_bias_bins=6,
    )


def labels() -> list[str]:
    """Recover FoodReg labels in the same ordering used by the data loader."""
    train_path = PROJECT / "preprocess/outputs/food/train.jsonlines"
    found: set[str] = set()
    for line in train_path.read_text(encoding="utf-8").splitlines():
        found.update(item["entity_type"] for item in json.loads(line)["entity_mentions"])
    return sorted(found)


def find_case(dataloader):
    for batch in dataloader:
        for row, words in enumerate(batch["raw_words"]):
            if "".join(words) == CASE_TEXT:
                return batch, row
    raise RuntimeError("The selected FoodReg test clause could not be found.")


def term_spans(tokens: list[str]) -> list[tuple[int, int]]:
    text = "".join(tokens)
    spans: list[tuple[int, int]] = []
    for _, phrase in UNITS:
        start = text.find(phrase)
        if start < 0:
            raise RuntimeError(f"Phrase not found in the selected FoodReg clause: {phrase}")
        spans.append((start, start + len(phrase) - 1))
    return spans


@torch.no_grad()
def forward_span_features(model, batch, device: torch.device):
    """Return final pre-classifier span features and logits for one sentence."""
    captured = []

    def hook(_module, inputs, _output):
        captured.append(inputs[0].detach().cpu())

    handle = model.score_head.register_forward_hook(hook)
    try:
        device_batch = move_to_device(batch, device)
        output = model(
            input_ids=device_batch["input_ids"],
            bpe_len=device_batch["bpe_len"],
            indexes=device_batch["indexes"],
            matrix=device_batch["matrix"],
            raw_words=device_batch["raw_words"],
        )["scores"].detach().cpu()
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("Failed to capture the final span representation.")
    return captured[0], output


def similarity_matrix(features: torch.Tensor, spans: list[tuple[int, int]]) -> list[list[float]]:
    span_vectors = torch.stack([features[start, end] for start, end in spans])
    span_vectors = F.normalize(span_vectors.float(), p=2, dim=-1, eps=1e-8)
    full_matrix = span_vectors @ span_vectors.T
    selected = full_matrix[:, list(ANCHOR_INDICES)]
    return [[float(value) for value in row] for row in selected]


def model_result(
    checkpoint: Path,
    use_snsa: bool,
    batch,
    matrix_segs,
    device: torch.device,
    row: int,
    iv_label: int,
    gold_iv_span: tuple[int, int],
    spans: list[tuple[int, int]],
):
    model = build_model(MODEL_NAME, matrix_segs, build_args(checkpoint, use_snsa)).to(device).eval()
    features, scores = forward_span_features(model, batch, device)
    # The project evaluator applies sigmoid first, then averages the two
    # directional span probabilities. Use that exact rule in this annotation.
    probabilities = scores.sigmoid()
    probabilities = (probabilities + probabilities.transpose(1, 2)) / 2
    probability = probabilities[row, gold_iv_span[0], gold_iv_span[1], iv_label].item()
    matrix = similarity_matrix(features[row], spans)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return matrix, probability


def load_font(size: int, bold: bool = False):
    """Prefer Times New Roman; fall back to its installed serif-compatible face.

    The workspace currently does not ship a Microsoft Times New Roman binary.
    STIX General is an installed Times-style serif face.  Adding a licensed
    Times New Roman TTF to ``fonts/`` automatically takes precedence.
    """
    font_dir = Path(__file__).with_name("fonts")
    candidates = (
        [
            font_dir / "Times New Roman Bold.ttf",
            font_dir / "timesbd.ttf",
            "/usr/local/lib/python3.12/dist-packages/matplotlib/mpl-data/fonts/ttf/STIXGeneralBol.ttf",
            "/usr/local/lib/python3.12/dist-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSerif-Bold.ttf",
        ]
        if bold
        else [
            font_dir / "Times New Roman.ttf",
            font_dir / "times.ttf",
            "/usr/local/lib/python3.12/dist-packages/matplotlib/mpl-data/fonts/ttf/STIXGeneral.ttf",
            "/usr/local/lib/python3.12/dist-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSerif.ttf",
        ]
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def colour(value: float, minimum: float = 0.20, maximum: float = 1.00) -> tuple[int, int, int]:
    """Blue-white-red colour scale matching the supplied heatmap convention."""
    ratio = min(1.0, max(0.0, (value - minimum) / (maximum - minimum)))
    low = (63, 123, 180)
    mid = (252, 246, 239)
    high = (157, 0, 43)
    if ratio <= 0.5:
        inner = ratio * 2
        return tuple(round(low[i] + (mid[i] - low[i]) * inner) for i in range(3))
    inner = (ratio - 0.5) * 2
    return tuple(round(mid[i] + (high[i] - mid[i]) * inner) for i in range(3))


def draw_panel(draw, matrix, title, x0: int, y0: int, panel_width: int):
    row_font = load_font(21)
    col_font = load_font(20, bold=True)
    value_font = load_font(19)
    title_font = load_font(25, bold=True)
    labels_font = load_font(17)
    rows = [name for name, _ in UNITS]
    columns = ("IN\ncondition", "Gold IV\nspan")
    cell_w, cell_h = 150, 68
    grid_x, grid_y = x0 + 205, y0 + 60
    draw.text((x0 + 12, y0), title, fill=(25, 31, 38), font=title_font)
    for col, label in enumerate(columns):
        box = draw.multiline_textbbox((0, 0), label, font=col_font, spacing=0)
        text_w = box[2] - box[0]
        draw.multiline_text(
            (grid_x + col * cell_w + (cell_w - text_w) / 2, grid_y - 49),
            label,
            fill=(36, 42, 48),
            font=col_font,
            align="center",
            spacing=0,
        )
    for row, label in enumerate(rows):
        box = draw.textbbox((0, 0), label, font=row_font)
        draw.text((grid_x - 15 - (box[2] - box[0]), grid_y + row * cell_h + 20), label, fill=(30, 36, 42), font=row_font)
        for col, value in enumerate(matrix[row]):
            x = grid_x + col * cell_w
            y = grid_y + row * cell_h
            draw.rectangle((x, y, x + cell_w, y + cell_h), fill=colour(value), outline=(246, 246, 246), width=2)
            number = f"{value:.2f}" if value < 0.995 else "1.00"
            number_box = draw.textbbox((0, 0), number, font=value_font)
            text_colour = (255, 255, 255) if value >= 0.71 or value <= 0.34 else (30, 35, 40)
            draw.text(
                (x + (cell_w - (number_box[2] - number_box[0])) / 2, y + (cell_h - (number_box[3] - number_box[1])) / 2 - 2),
                number,
                fill=text_colour,
                font=value_font,
            )
    draw.text((grid_x, grid_y + len(rows) * cell_h + 16), "Anchor spans are the two columns; all row spans are from the same rule.", fill=(87, 101, 115), font=labels_font)
    return grid_y + len(rows) * cell_h


def draw_colorbar(draw, x: int, y: int, height: int):
    tick_font = load_font(15)
    for step in range(height):
        value = 1.0 - (step / max(height - 1, 1)) * 0.8
        draw.line((x, y + step, x + 24, y + step), fill=colour(value), width=1)
    draw.rectangle((x, y, x + 24, y + height), outline=(55, 55, 55), width=1)
    for value in (1.00, 0.80, 0.60, 0.40, 0.20):
        position = y + (1.0 - value) / 0.8 * height
        draw.line((x + 24, position, x + 31, position), fill=(55, 55, 55), width=1)
        draw.text((x + 38, position - 9), f"{value:.2f}", fill=(50, 57, 63), font=tick_font)
    draw.text((x - 8, y + height + 18), "cosine similarity", fill=(70, 81, 92), font=tick_font)


def render(output_dir: Path, no_snsa_matrix, full_matrix, no_snsa_prob: float, full_prob: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1870, 1010), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(35, bold=True)
    subtitle_font = load_font(20)
    caption_font = load_font(18)
    small_font = load_font(16)

    draw.text((55, 30), "Semantic similarity visualization for a FoodReg nutrient-content rule", fill=(20, 28, 35), font=title_font)
    draw.text(
        (55, 83),
        "Cosine similarity of real span features immediately before the final classifier; the Chinese rule is translated below.",
        fill=(88, 103, 117),
        font=subtitle_font,
    )
    draw.rounded_rectangle((55, 126, 1810, 205), radius=13, fill=(248, 249, 251), outline=(208, 216, 224), width=2)
    draw.text(
        (79, 146),
        "Rule: reduce sugar — compared with a reference food, sugar content is reduced by at least 25%.",
        fill=(37, 49, 60),
        font=caption_font,
    )
    draw.text(
        (79, 176),
        "The exact IV span is retained by Full SPSR-Net but missed by w/o SNSA at the same decision threshold.",
        fill=(82, 97, 110),
        font=small_font,
    )

    bottom = draw_panel(draw, no_snsa_matrix, "(a) SPSR-Net w/o SNSA", 55, 245, 760)
    draw_panel(draw, full_matrix, "(b) Full SPSR-Net", 875, 245, 760)
    draw_colorbar(draw, 1695, 310, 470)

    draw.line((55, bottom + 74, 1810, bottom + 74), fill=(213, 220, 227), width=2)
    draw.text(
        (55, bottom + 99),
        "Observed effect: without SNSA, the gold composite IV is more similar to its individual fragments; Full SPSR-Net preserves a more distinct composite-span representation.",
        fill=(44, 61, 74),
        font=caption_font,
    )
    draw.text(
        (55, bottom + 132),
        f"Exact gold IV probability: w/o SNSA = {no_snsa_prob:.3f} < τ = {THRESHOLD:.2f};  Full SPSR-Net = {full_prob:.3f} > τ = {THRESHOLD:.2f}.",
        fill=(44, 61, 74),
        font=caption_font,
    )
    draw.text(
        (55, bottom + 166),
        "This is a qualitative case study based on one FoodReg test clause, not an aggregate performance measure.",
        fill=(100, 112, 124),
        font=small_font,
    )
    image.save(output_dir / "Figure3_FoodReg_semantic_similarity.png", dpi=(300, 300))
    image.save(output_dir / "Figure3_FoodReg_semantic_similarity.pdf", resolution=300.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use the same batch size as the FoodReg probability export so the
    # displayed decision follows the paper's inference configuration exactly.
    loaders, matrix_segs = load_data(MODEL_NAME, "food", batch_size=4, num_workers=1)
    batch, row = find_case(loaders["test"])
    tokens = list(batch["raw_words"][row])[: int(batch["word_len"][row])]
    spans = term_spans(tokens)
    label_names = labels()
    iv_index = label_names.index("IV")
    gold_iv_span = spans[6]

    no_snsa_matrix, no_snsa_probability = model_result(
        NO_SNSA_CHECKPOINT, False, batch, matrix_segs, device, row, iv_index, gold_iv_span, spans
    )
    full_matrix, full_probability = model_result(
        FULL_CHECKPOINT, True, batch, matrix_segs, device, row, iv_index, gold_iv_span, spans
    )
    render(args.output_dir, no_snsa_matrix, full_matrix, no_snsa_probability, full_probability)

    report = {
        "case_text": CASE_TEXT,
        "translated_rule": "Reduce sugar: compared with a reference food, sugar content is reduced by at least 25%.",
        "units": [{"name": name, "text": text, "span": list(span)} for (name, text), span in zip(UNITS, spans)],
        "anchor_units": [UNITS[index][0] for index in ANCHOR_INDICES],
        "matrices": {"w/o SNSA": no_snsa_matrix, "Full SPSR-Net": full_matrix},
        "exact_gold_iv_probability": {"w/o SNSA": no_snsa_probability, "Full SPSR-Net": full_probability},
        "threshold": THRESHOLD,
        "checkpoints": {"w/o SNSA": str(NO_SNSA_CHECKPOINT), "Full SPSR-Net": str(FULL_CHECKPOINT)},
        "feature_source": "Input to score_head (span representation immediately before final classification).",
    }
    report_path = args.output_dir / "Figure3_FoodReg_semantic_similarity_data.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figure": str(args.output_dir / "Figure3_FoodReg_semantic_similarity.png"), "report": str(report_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
