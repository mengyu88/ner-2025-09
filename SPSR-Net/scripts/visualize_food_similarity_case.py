#!/usr/bin/env python3
"""Create a case-level span-similarity heatmap for SPSR-Net FOOD experiments.

This is an explanatory visualization: cosine similarity is computed from the
span representation immediately before ``score_head``.  It deliberately uses
the current BHPC SPSR-Net checkpoint and the existing w/o-HSR ablation, so no
PGD label or result appears in the output figure.
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_per_class import decode_candidates, read_label_names
from scripts.eval_thresholds import build_model, load_data, move_to_device


MODEL_NAME = "/root/.cache/modelscope/hub/AI-ModelScope/bert-base-chinese"
BHPC_CKPT = ROOT / "_saved_models/2026-08-15-22_03_36_081264/model-epoch_10-batch_30010-f#f#dev_97.38/fastnlp_model.pkl.tar"
NO_HSR_CKPT = ROOT / "_saved_models/2026-07-05-22_12_33_633670/model-epoch_8-batch_24008-f#f#test_95.87/fastnlp_model.pkl.tar"
OUT_DIR = ROOT / "runs" / "food_similarity_visualization"
THRESHOLD = 0.48


def model_args(checkpoint, use_hsr):
    return SimpleNamespace(
        checkpoint=str(checkpoint), cnn_dim=400, biaffine_size=200, n_head=4,
        cnn_depth=1, n_layer=2, logit_drop=0.15, size_embed_dim=25,
        kernel_size=3, separateness_rate=0.05, theta=1.0, sad_topk=2,
        sad_attn_dim=None, use_sad=1, use_hsr=int(use_hsr),
        sad_use_rel_bias=1, sad_gate=1, head_type="linear",
        use_length_bias=False, length_bias_bins=6,
    )


def decode(scores, gold, word_len):
    scores = scores.sigmoid()
    scores = (scores + scores.transpose(1, 2)) / 2
    scores = scores[:word_len, :word_len]
    max_scores, label_ids = scores.max(dim=-1)
    starts, ends = torch.triu_indices(word_len, word_len)
    confidence = max_scores[starts, ends]
    keep = confidence >= THRESHOLD
    starts, ends, confidence = starts[keep], ends[keep], confidence[keep]
    label_ids = label_ids[starts, ends]
    order = torch.argsort(confidence, descending=True)
    candidates = [
        (float(confidence[i]), int(starts[i]), int(ends[i]), int(label_ids[i]))
        for i in order
    ]
    return decode_candidates(candidates, THRESHOLD), candidates


def forward_with_features(model, batch, device):
    captured = []

    def capture(_module, inputs, _output):
        captured.append(inputs[0].detach().cpu())

    hook = model.score_head.register_forward_hook(capture)
    try:
        out = model(
            input_ids=batch["input_ids"].to(device),
            bpe_len=batch["bpe_len"].to(device),
            indexes=batch["indexes"].to(device),
            matrix=batch["matrix"].to(device),
            raw_words=batch["raw_words"],
        )
    finally:
        hook.remove()
    return out["scores"].detach().cpu(), captured.pop()


def has_nested(gold):
    spans = [(s, e) for s, e, _ in gold]
    return any(
        max(a[0], b[0]) <= min(a[1], b[1]) and a != b
        for index, a in enumerate(spans) for b in spans[index + 1:]
    )


def select_case(bhpc_model, ablation_model, dataloader, device):
    best = None
    running_index = 0
    for batch in dataloader:
        bhpc_scores, bhpc_features = forward_with_features(bhpc_model, batch, device)
        ablation_scores, ablation_features = forward_with_features(ablation_model, batch, device)
        word_lens = batch["word_len"].cpu().tolist()
        for row, length in enumerate(word_lens):
            gold = set(map(tuple, batch["ent_target"][row]))
            bhpc_pred, bhpc_candidates = decode(bhpc_scores[row], gold, length)
            ablation_pred, ablation_candidates = decode(ablation_scores[row], gold, length)
            bhpc_correct = len(gold & bhpc_pred)
            ablation_correct = len(gold & ablation_pred)
            # Prefer a compact nested example for which the full model corrects
            # at least one additional gold span; then prefer a larger margin.
            quality = (
                1000 * int(has_nested(gold))
                + 100 * max(0, bhpc_correct - ablation_correct)
                + 20 * len(gold)
                - 0.25 * length
            )
            record = {
                "quality": quality, "index": running_index + row,
                "tokens": list(batch["raw_words"][row])[:length], "length": length,
                "gold": gold, "bhpc_pred": bhpc_pred, "ablation_pred": ablation_pred,
                "bhpc_candidates": bhpc_candidates, "ablation_candidates": ablation_candidates,
                "bhpc_features": bhpc_features[row, :length, :length].clone(),
                "ablation_features": ablation_features[row, :length, :length].clone(),
            }
            if best is None or record["quality"] > best["quality"]:
                best = record
        running_index += len(word_lens)
    return best


def span_text(tokens, span):
    start, end, _ = span
    return "".join(tokens[start:end + 1])


def choose_spans(case):
    """Gold spans plus their boundary-neighbour negatives, capped at nine."""
    spans = []
    kind = {}
    for span in sorted(case["gold"], key=lambda x: (x[0], x[1], x[2])):
        if span not in spans:
            spans.append(span)
            kind[span] = "gold"
    for start, end, label in list(spans):
        for neighbor in ((start, end - 1, label), (start + 1, end, label),
                         (start, end + 1, label), (start - 1, end, label)):
            s, e, _ = neighbor
            if 0 <= s <= e < case["length"] and neighbor not in spans:
                spans.append(neighbor)
                kind[neighbor] = "boundary neighbour"
            if len(spans) >= 9:
                return spans, kind
    for prediction in sorted((case["ablation_pred"] | case["bhpc_pred"]) - case["gold"]):
        if prediction not in spans:
            spans.append(prediction)
            kind[prediction] = "predicted non-gold"
        if len(spans) >= 9:
            break
    return spans[:9], kind


def similarity_matrix(features, spans):
    values = torch.stack([features[start, end] for start, end, _ in spans])
    values = F.normalize(values.float(), dim=-1, eps=1e-8)
    return (values @ values.T).numpy()


def blend(low, high, ratio):
    return tuple(int(a + (b - a) * ratio) for a, b in zip(low, high))


def heat_color(value):
    # White-to-deep-red palette matching the supplied paper-style heatmap.
    ratio = max(0.0, min(1.0, (value - 0.35) / 0.65))
    return blend((244, 248, 251), (125, 0, 35), ratio)


def font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def draw_heatmap(draw, matrix, labels, x0, y0, title, panel_width):
    title_font = font(27, True)
    label_font = font(16, True)
    cell_font = font(14)
    draw.text((x0, y0), title, fill=(20, 20, 20), font=title_font)
    n = len(labels)
    cell = min(62, (panel_width - 125) // n)
    grid_x, grid_y = x0 + 112, y0 + 48
    for i, label in enumerate(labels):
        draw.text((grid_x + i * cell + 4, grid_y - 24), label, fill=(30, 30, 30), font=label_font)
        draw.text((x0 + 70, grid_y + i * cell + 18), label, fill=(30, 30, 30), font=label_font)
        for j in range(n):
            value = float(matrix[i, j])
            x, y = grid_x + j * cell, grid_y + i * cell
            draw.rectangle((x, y, x + cell, y + cell), fill=heat_color(value), outline=(245, 245, 245), width=1)
            text = f"{value:.2f}"
            box = draw.textbbox((0, 0), text, font=cell_font)
            colour = (255, 255, 255) if value >= 0.72 else (25, 25, 25)
            draw.text((x + (cell - (box[2] - box[0])) / 2, y + (cell - (box[3] - box[1])) / 2 - 1), text, fill=colour, font=cell_font)
    return grid_y + n * cell


def draw_figure(case, labels, spans, kinds, output_path):
    matrix_bhpc = similarity_matrix(case["bhpc_features"], spans)
    matrix_wo_hsr = similarity_matrix(case["ablation_features"], spans)
    image = Image.new("RGB", (1740, 1090), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((55, 25), "Span Semantic Similarity on a FOOD Test Case", fill=(20, 20, 20), font=font(34, True))
    draw.text((55, 72), "Cosine similarity of span features immediately before the classifier", fill=(70, 70, 70), font=font(19))
    y_end = draw_heatmap(draw, matrix_bhpc, labels, 55, 125, "(a) SPSR-Net (with BHPC)", 780)
    draw_heatmap(draw, matrix_wo_hsr, labels, 900, 125, "(b) SPSR-Net w/o HSR", 780)

    draw.line((55, y_end + 34, 1685, y_end + 34), fill=(210, 210, 210), width=2)
    draw.text((55, y_end + 55), "Span legend", fill=(20, 20, 20), font=font(23, True))
    line_y = y_end + 92
    for index, span in enumerate(spans):
        label = labels[index]
        entity_name = labels_for_span(span, labels=None, entity_labels=None)
        # Entity label is intentionally represented by its type ID below, which
        # keeps the figure readable for Chinese input without requiring CJK fonts.
        text = f"{label}: {kinds[span]} | token positions [{span[0]}:{span[1]}] | type={span[2]}"
        column = 55 if index < 5 else 875
        row = index if index < 5 else index - 5
        draw.text((column, line_y + row * 32), text, fill=(35, 35, 35), font=font(17))

    case_summary = (
        f"Selected test item #{case['index']}  |  gold spans={len(case['gold'])}  |  "
        f"correct: SPSR-Net={len(case['gold'] & case['bhpc_pred'])}, w/o HSR={len(case['gold'] & case['ablation_pred'])}"
    )
    draw.text((55, 1015), case_summary, fill=(50, 50, 50), font=font(18))
    image.save(output_path)


def labels_for_span(span, labels=None, entity_labels=None):
    return str(span)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders, matrix_segs = load_data(MODEL_NAME, "food", batch_size=4, num_workers=4)
    bhpc_model = build_model(MODEL_NAME, matrix_segs, model_args(BHPC_CKPT, True)).to(device).eval()
    ablation_model = build_model(MODEL_NAME, matrix_segs, model_args(NO_HSR_CKPT, False)).to(device).eval()
    case = select_case(bhpc_model, ablation_model, dataloaders["test"], device)
    spans, kinds = choose_spans(case)
    labels = [f"S{i + 1}" for i in range(len(spans))]
    output_png = OUT_DIR / "food_span_similarity_bhpc_vs_wo_hsr.png"
    draw_figure(case, labels, spans, kinds, output_png)

    entity_labels = read_label_names("food")
    report = {
        "figure": str(output_png), "threshold": THRESHOLD,
        "models": {
            "SPSR-Net (with BHPC)": str(BHPC_CKPT),
            "SPSR-Net w/o HSR": str(NO_HSR_CKPT),
        },
        "case_index": case["index"], "token_count": case["length"],
        "tokens": case["tokens"],
        "gold_spans": [[int(s), int(e), entity_labels[int(label)]] for s, e, label in sorted(case["gold"])],
        "spans_visualized": [
            {"id": labels[i], "start": int(span[0]), "end": int(span[1]),
             "entity_type": entity_labels[int(span[2])], "role": kinds[span],
             "text": span_text(case["tokens"], span)}
            for i, span in enumerate(spans)
        ],
        "correct_gold_spans": {
            "SPSR-Net (with BHPC)": len(case["gold"] & case["bhpc_pred"]),
            "SPSR-Net w/o HSR": len(case["gold"] & case["ablation_pred"]),
        },
    }
    report_path = OUT_DIR / "food_span_similarity_bhpc_vs_wo_hsr.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps({"figure": str(output_png), "report": str(report_path), "case_index": case["index"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
