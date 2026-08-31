"""Generate three FoodReg semantic-similarity figure layouts.

Each panel follows the logic of the supplied SRT visualization: a target span
is fixed for one real test case; rows are semantically relevant spans from
that same clause; columns are model variants.  The values are cosine
similarities between real span representations immediately before the final
classifier.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).parent
PROJECT = Path("/root/shared-nvme/projects/SPSR-Net")
for entry in (SCRIPT_DIR, PROJECT, PROJECT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from eval_thresholds import build_model, load_data, move_to_device  # noqa: E402
from plot_figure3_food_semantic_similarity import colour, load_font  # noqa: E402


MODEL_NAME = "/root/.cache/modelscope/hub/AI-ModelScope/bert-base-chinese"
CHECKPOINTS = {
    "Full SPSR-Net": (
        PROJECT / "_saved_models/2026-07-03-00_07_48_429272/"
        "model-epoch_15-batch_45015-f#f#test_97.98/fastnlp_model.pkl.tar",
        True,
        True,
    ),
    "w/o SNSA": (
        PROJECT / "_saved_models/2026-07-04-20_43_24_622169/"
        "model-epoch_6-batch_18006-f#f#test_97.13/fastnlp_model.pkl.tar",
        False,
        True,
    ),
    "w/o HSR": (
        PROJECT / "_saved_models/2026-07-05-22_12_33_633670/"
        "model-epoch_8-batch_24008-f#f#test_95.87/fastnlp_model.pkl.tar",
        True,
        False,
    ),
}


@dataclass(frozen=True)
class Unit:
    label: str
    phrase: str
    within_target: bool = False
    occurrence: int = 0


@dataclass(frozen=True)
class Case:
    key: str
    text: str
    target: str
    units: tuple[Unit, ...]


MINERAL_CLAIM = Case(
    key="mineral_claim",
    text=(
        "增加矿物质(不包括钠)的指标值为与参考食品比较，矿物质含量增加25%以上(含25%)，"
        "检验方法为参考食品的数据来源：1.同一企业同类或同一属类食品的营养成分含量或2.《中国食物成分表》中同类食品营养成分含量。"
    ),
    target="与参考食品比较，矿物质含量增加25%以上(含25%)",
    units=(
        Unit("complete IV", "与参考食品比较，矿物质含量增加25%以上(含25%)"),
        Unit("comparison", "与参考食品比较", True),
        Unit("reference food", "参考食品", True),
        Unit("mineral", "矿物质", True),
        Unit("increase", "含量增加", True),
        Unit("threshold ≥25%", "25%以上", True),
    ),
)

SUGAR_CLAIM = Case(
    key="sugar_claim",
    text=(
        "减少糖的指标值为与参考食品比较，糖含量减少25%以上，"
        "检验方法为参考食品(基准食品)应为消费者熟知、容易理解的同类或同一属类食品。"
    ),
    target="与参考食品比较，糖含量减少25%以上",
    units=(
        Unit("complete IV", "与参考食品比较，糖含量减少25%以上"),
        Unit("comparison", "与参考食品比较", True),
        Unit("reference food", "参考食品", True),
        Unit("sugar", "糖", True),
        Unit("reduction", "含量减少", True),
        Unit("threshold ≥25%", "25%以上", True),
    ),
)

BEVERAGE_DEFINITION = Case(
    key="beverage_definition",
    text=(
        "用一种或几种食用原料，添加或不添加辅料、食品添加剂、食品营养强化剂，经加工制成定量包装的、"
        "供直接饮用或冲调饮用、乙醇含量不超过质量分数为0.5%的制品，也可称为饮品，如碳酸饮料、果蔬汁类及其饮料、蛋白饮料、固体饮料等。"
    ),
    target="食品添加剂",
    units=(
        Unit("food additive", "食品添加剂"),
        Unit("nutrient fortifier", "营养强化剂"),
        Unit("carbonated beverage", "饮料", occurrence=0),
        Unit("fruit/vegetable drink", "饮料", occurrence=1),
        Unit("protein drink", "饮料", occurrence=2),
        Unit("solid drink", "饮料", occurrence=3),
    ),
)

OIL_CATEGORY = Case(
    key="oil_category",
    text=(
        "6.7.1.2对于稠厚或半稠厚制品以及难以从中分出汁液的制品（如：糖浆、果酱、果冻、油脂等），"
        "取一部分样品在均质器或研钵中研磨，如果研磨后的样品仍太稠厚，加入等量的无菌蒸馏水，混匀备用。"
    ),
    target="油脂",
    units=(
        Unit("oil/fat", "油脂"),
        Unit("viscous product", "稠厚或半稠厚制品"),
        Unit("syrup", "糖浆"),
        Unit("jam", "果酱"),
        Unit("jelly", "果冻"),
        Unit("sample", "样品"),
        Unit("grinding", "研磨"),
        Unit("sterile water", "无菌蒸馏水"),
    ),
)

CONTAMINANT_STANDARD = Case(
    key="contaminant_standard",
    text="GB2762—2017《食品安全国家标准食品中污染物限量》国家标准第1号修改单",
    target="GB2762—2017",
    units=(
        Unit("GB2762—2017", "GB2762—2017"),
        Unit("food-safety standard", "食品安全国家标准"),
        Unit("food", "食品"),
        Unit("contaminant", "污染物"),
        Unit("limit", "限量"),
        Unit("national standard", "国家标准"),
        Unit("amendment No. 1", "第1号修改单"),
    ),
)


def build_args(checkpoint: Path, use_snsa: bool, use_hsr: bool) -> SimpleNamespace:
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
        names.update(item["entity_type"] for item in json.loads(line)["entity_mentions"])
    return sorted(names)


def locate_cases(dataloader, cases: tuple[Case, ...]):
    wanted = {case.text: case for case in cases}
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
        raise RuntimeError(f"Unable to find {len(missing)} selected FoodReg test cases.")
    return {case.key: (case, *found[case.text]) for case in cases}


def span_locations(case: Case) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    target_start = case.text.find(case.target)
    if target_start < 0:
        raise RuntimeError(f"Target span missing from case: {case.key}")
    target = (target_start, target_start + len(case.target) - 1)
    spans = []
    for unit in case.units:
        search_text = case.target if unit.within_target else case.text
        offset = target_start if unit.within_target else 0
        start = -1
        search_from = 0
        for _ in range(unit.occurrence + 1):
            start = search_text.find(unit.phrase, search_from)
            if start < 0:
                break
            search_from = start + len(unit.phrase)
        if start < 0:
            raise RuntimeError(f"Unit '{unit.phrase}' missing from case: {case.key}")
        spans.append((offset + start, offset + start + len(unit.phrase) - 1))
    return target, spans


@torch.no_grad()
def forward_features(model, batch, device: torch.device):
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
        raise RuntimeError("Failed to capture the pre-classifier span features.")
    return captured[0], logits


def collect_model_results(model_name, selected, matrix_segs, device: torch.device):
    checkpoint, use_snsa, use_hsr = CHECKPOINTS[model_name]
    model = build_model(MODEL_NAME, matrix_segs, build_args(checkpoint, use_snsa, use_hsr)).to(device).eval()
    results = {}
    for case, batch, row in selected.values():
        features, logits = forward_features(model, batch, device)
        target, unit_spans = span_locations(case)
        target_vector = F.normalize(features[row, target[0], target[1]].float(), dim=-1, eps=1e-8)
        unit_vectors = F.normalize(
            torch.stack([features[row, start, end] for start, end in unit_spans]).float(), dim=-1, eps=1e-8
        )
        similarities = (unit_vectors @ target_vector).tolist()
        probabilities = logits.sigmoid()
        probabilities = (probabilities + probabilities.transpose(1, 2)) / 2
        matching_labels = [
            int(label)
            for start, end, label in batch["ent_target"][row]
            if int(start) == target[0] and int(end) == target[1]
        ]
        if len(matching_labels) != 1:
            raise RuntimeError(f"Target is not a unique FoodReg gold entity: {case.key}")
        target_label = matching_labels[0]
        results[case.key] = {
            "similarities": [float(value) for value in similarities],
            "unit_probabilities": [
                float(probabilities[row, start, end, target_label]) for start, end in unit_spans
            ],
            "target_probability": float(probabilities[row, target[0], target[1], target_label]),
            "target_label_id": target_label,
            "target_span": list(target),
            "unit_spans": [list(span) for span in unit_spans],
        }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def panel_range(values: list[float]) -> tuple[float, tuple[float, ...]]:
    minimum = max(0.0, min(values))
    minimum = max(0.0, (int(minimum * 10) / 10) - 0.1)
    minimum = min(minimum, 0.5)
    ticks = tuple(round(minimum + index * (1.0 - minimum) / 5, 2) for index in range(6))
    return minimum, ticks


def draw_colourbar(draw, x: int, y: int, height: int, minimum: float, ticks: tuple[float, ...]):
    tick_font = load_font(10)
    for pixel in range(height):
        value = 1.0 - pixel / max(1, height - 1) * (1.0 - minimum)
        draw.line((x, y + pixel, x + 14, y + pixel), fill=colour(value, minimum=minimum, maximum=1.0), width=1)
    draw.rectangle((x, y, x + 14, y + height), outline=(70, 70, 70), width=1)
    for value in reversed(ticks):
        position = y + (1.0 - value) / (1.0 - minimum) * height
        draw.line((x + 14, position, x + 18, position), fill=(65, 65, 65), width=1)
        draw.text((x + 23, position - 6), f"{value:.1f}", fill=(55, 62, 68), font=tick_font)


def draw_panel(draw, case: Case, model_names, results, x0: int, y0: int, caption: str):
    """Draw one reference-style case panel and return its right/bottom edge."""
    row_font = load_font(13)
    column_font = load_font(13, True)
    value_font = load_font(12)
    caption_font = load_font(15)
    cell_w, cell_h, label_width = 140, 28, 190
    grid_x, grid_y = x0 + label_width, y0
    all_values = [value for name in model_names for value in results[name][case.key]["similarities"]]
    minimum, ticks = panel_range(all_values)
    for column, model_name in enumerate(model_names):
        column_label = {"w/o SNSA": "w/o SNSA", "w/o HSR": "w/o HSR", "Full SPSR-Net": "Full"}[model_name]
        text_box = draw.textbbox((0, 0), column_label, font=column_font)
        draw.text(
            (grid_x + column * cell_w + (cell_w - (text_box[2] - text_box[0])) / 2, grid_y - 25),
            column_label,
            fill=(24, 29, 34),
            font=column_font,
        )
    for row, unit in enumerate(case.units):
        label_box = draw.textbbox((0, 0), unit.label, font=row_font)
        draw.text((grid_x - 12 - (label_box[2] - label_box[0]), grid_y + row * cell_h + 7), unit.label, fill=(30, 35, 40), font=row_font)
        for column, model_name in enumerate(model_names):
            value = results[model_name][case.key]["similarities"][row]
            x, y = grid_x + column * cell_w, grid_y + row * cell_h
            draw.rectangle((x, y, x + cell_w, y + cell_h), fill=colour(value, minimum=minimum, maximum=1.0), outline=(248, 248, 248), width=1)
            number = "1" if value >= 0.995 else f"{value:.2f}"
            box = draw.textbbox((0, 0), number, font=value_font)
            foreground = (255, 255, 255) if value >= minimum + (1.0 - minimum) * 0.62 else (29, 33, 37)
            draw.text((x + (cell_w - (box[2] - box[0])) / 2, y + (cell_h - (box[3] - box[1])) / 2 - 1), number, fill=foreground, font=value_font)
    grid_width, grid_height = len(model_names) * cell_w, len(case.units) * cell_h
    caption_box = draw.textbbox((0, 0), caption, font=caption_font)
    draw.text((grid_x + grid_width / 2 - (caption_box[2] - caption_box[0]) / 2, grid_y + grid_height + 17), caption, fill=(25, 31, 36), font=caption_font)
    draw_colourbar(draw, grid_x + grid_width + 13, grid_y, grid_height, minimum, ticks)
    return grid_x + grid_width + 75, grid_y + grid_height + 40


def new_canvas(width: int, height: int):
    return Image.new("RGB", (width, height), "white"), width, height


def save(image: Image.Image, directory: Path, stem: str):
    image.save(directory / f"{stem}.png", dpi=(300, 300))
    image.save(directory / f"{stem}.pdf", resolution=300.0)


def render_figures(output_dir: Path, results):
    output_dir.mkdir(parents=True, exist_ok=True)
    # Figure 1: reference-style dual panel. Each panel isolates one module
    # on a different FoodReg case, with the ablation on the left and Full on
    # the right.
    image, _, _ = new_canvas(1400, 300)
    draw = ImageDraw.Draw(image)
    draw_panel(
        draw,
        BEVERAGE_DEFINITION,
        ("w/o SNSA", "Full SPSR-Net"),
        results,
        50,
        45,
        "(a) Beverage-definition case (FC)",
    )
    draw_panel(
        draw,
        MINERAL_CLAIM,
        ("w/o HSR", "Full SPSR-Net"),
        results,
        740,
        45,
        "(b) Mineral-increase claim (IV)",
    )
    save(image, output_dir, "Figure5_two_foodreg_module_cases")

    # Version 2: SNSA over two semantically distinct FoodReg cases.
    image, _, _ = new_canvas(1900, 350)
    draw = ImageDraw.Draw(image)
    draw_panel(
        draw,
        SUGAR_CLAIM,
        ("w/o SNSA", "Full SPSR-Net"),
        results,
        65,
        55,
        "(a) Nutrient-content claim (IV)",
    )
    draw_panel(
        draw,
        OIL_CATEGORY,
        ("w/o SNSA", "Full SPSR-Net"),
        results,
        1010,
        55,
        "(b) Food-category mention (FC)",
    )
    save(image, output_dir, "Figure6_SNSA_two_diverse_cases")

    # Version 3: HSR over two semantically distinct FoodReg cases.
    image, _, _ = new_canvas(1900, 350)
    draw = ImageDraw.Draw(image)
    draw_panel(
        draw,
        MINERAL_CLAIM,
        ("w/o HSR", "Full SPSR-Net"),
        results,
        65,
        55,
        "(a) Nutrient-content claim (IV)",
    )
    draw_panel(
        draw,
        CONTAMINANT_STANDARD,
        ("w/o HSR", "Full SPSR-Net"),
        results,
        1010,
        55,
        "(b) Contaminant-limit standard (SNum)",
    )
    save(image, output_dir, "Figure7_HSR_two_diverse_cases")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    args = parser.parse_args()
    selected_cases = (MINERAL_CLAIM, SUGAR_CLAIM, BEVERAGE_DEFINITION, OIL_CATEGORY, CONTAMINANT_STANDARD)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, matrix_segs = load_data(MODEL_NAME, "food", batch_size=4, num_workers=1)
    selected = locate_cases(loaders["test"], selected_cases)
    results = {
        model_name: collect_model_results(model_name, selected, matrix_segs, device)
        for model_name in CHECKPOINTS
    }
    render_figures(args.output_dir, results)
    report = {
        "models": {
            name: {"checkpoint": str(checkpoint), "use_snsa": use_snsa, "use_hsr": use_hsr}
            for name, (checkpoint, use_snsa, use_hsr) in CHECKPOINTS.items()
        },
        "cases": {
            case.key: {
                "text": case.text,
                "target": case.target,
                "units": [unit.__dict__ for unit in case.units],
                "results": {model_name: results[model_name][case.key] for model_name in CHECKPOINTS},
            }
            for case in selected_cases
        },
        "feature_source": "Input to score_head (span representation immediately before final classification).",
    }
    (args.output_dir / "FoodReg_similarity_three_layouts_data.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
