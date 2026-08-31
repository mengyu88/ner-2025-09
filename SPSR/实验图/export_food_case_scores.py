#!/usr/bin/env python3
"""Export real span-class probabilities for one FoodReg test sentence.

The output is intentionally limited to a selected FoodReg test case.  It is
used to create a qualitative case-study figure; it is not an aggregate metric.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT = Path("/root/shared-nvme/projects/SPSR-Net")
SCRIPTS = PROJECT / "scripts"
for item in (PROJECT, SCRIPTS):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from eval_thresholds import build_model, load_data, move_to_device  # noqa: E402


MODEL_NAME = "/root/.cache/modelscope/hub/AI-ModelScope/bert-base-chinese"
CASE_TEXTS = {
    "增加矿物质(不包括钠)的指标值为与参考食品比较，矿物质含量增加25%以上(含25%)，"
    "检验方法为参考食品的数据来源：1.同一企业同类或同一属类食品的营养成分含量或2.《中国食物成分表》中同类食品营养成分含量。",
}


def build_args(checkpoint: str, use_sad: int, use_hsr: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint=checkpoint,
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
        use_sad=use_sad,
        use_hsr=use_hsr,
        sad_use_rel_bias=1,
        sad_gate=1,
        head_type="linear",
        use_length_bias=False,
        length_bias_bins=6,
    )


def read_labels() -> list[str]:
    labels = set()
    path = PROJECT / "preprocess/outputs/food/train.jsonlines"
    for line in path.read_text(encoding="utf-8").splitlines():
        labels.update(entity["entity_type"] for entity in json.loads(line)["entity_mentions"])
    return sorted(labels)


@torch.no_grad()
def export_cases(
    checkpoint: str,
    use_sad: int,
    name: str,
    device: torch.device,
    use_hsr: int = 1,
) -> list[dict]:
    # The project dataloader sets a prefetch factor, so it requires at least
    # one worker even though this script only exports one example.
    loaders, matrix_segs = load_data(MODEL_NAME, "food", batch_size=4, num_workers=1)
    model = build_model(MODEL_NAME, matrix_segs, build_args(checkpoint, use_sad, use_hsr)).to(device)
    model.eval()
    labels = read_labels()
    found = []
    remaining = set(CASE_TEXTS)
    for batch in loaders["test"]:
        raw_batch = batch["raw_words"]
        batch_on_device = move_to_device(batch, device)
        output = model(
            input_ids=batch_on_device["input_ids"],
            bpe_len=batch_on_device["bpe_len"],
            indexes=batch_on_device["indexes"],
            matrix=batch_on_device["matrix"],
            raw_words=batch_on_device["raw_words"],
        )["scores"].detach().cpu()
        probs = output.sigmoid()
        probs = (probs + probs.transpose(1, 2)) / 2
        for index, raw_words in enumerate(raw_batch):
            text = "".join(raw_words)
            if text not in remaining:
                continue
            length = len(raw_words)
            gold = [list(map(int, x)) for x in batch["ent_target"][index]]
            found.append({
                "model": name,
                "checkpoint": checkpoint,
                "use_snsa": bool(use_sad),
                "tokens": raw_words,
                "text": text,
                "labels": labels,
                "gold": [
                    {
                        "start": start,
                        "end": end,
                        "label": labels[label_id],
                        "text": "".join(raw_words[start:end + 1]),
                    }
                    for start, end, label_id in gold
                ],
                "probabilities": probs[index, :length, :length].tolist(),
            })
            remaining.remove(text)
        if not remaining:
            return found
    missing = " | ".join(sorted(remaining))
    raise RuntimeError(f"The requested FoodReg test case was not found: {missing}")


@torch.no_grad()
def profile_gold_scores(
    checkpoint: str,
    use_sad: int,
    name: str,
    device: torch.device,
    use_hsr: int = 1,
) -> list[dict]:
    """Capture exact-label probabilities of every gold test span.

    This compact profile is only used to identify a truthful qualitative
    example where the full model and its ablation differ. It does not export
    every model logit for the full test corpus.
    """
    loaders, matrix_segs = load_data(MODEL_NAME, "food", batch_size=4, num_workers=1)
    model = build_model(MODEL_NAME, matrix_segs, build_args(checkpoint, use_sad, use_hsr)).to(device)
    model.eval()
    labels = read_labels()
    profiles = []
    for batch in loaders["test"]:
        raw_batch = batch["raw_words"]
        batch_on_device = move_to_device(batch, device)
        logits = model(
            input_ids=batch_on_device["input_ids"],
            bpe_len=batch_on_device["bpe_len"],
            indexes=batch_on_device["indexes"],
            matrix=batch_on_device["matrix"],
            raw_words=batch_on_device["raw_words"],
        )["scores"].detach().cpu()
        probabilities = logits.sigmoid()
        probabilities = (probabilities + probabilities.transpose(1, 2)) / 2
        for index, raw_words in enumerate(raw_batch):
            gold = []
            for start, end, label_id in batch["ent_target"][index]:
                start, end, label_id = int(start), int(end), int(label_id)
                nearby = []
                for start_shift in range(-2, 3):
                    for end_shift in range(-2, 3):
                        if start_shift == 0 and end_shift == 0:
                            continue
                        shifted_start = start + start_shift
                        shifted_end = end + end_shift
                        if 0 <= shifted_start <= shifted_end < len(raw_words):
                            nearby.append({
                                "start_shift": start_shift,
                                "end_shift": end_shift,
                                "probability": float(
                                    probabilities[index, shifted_start, shifted_end, label_id]
                                ),
                            })
                strongest_nearby = max(nearby, key=lambda item: item["probability"], default=None)
                gold.append({
                    "start": start,
                    "end": end,
                    "label": labels[label_id],
                    "text": "".join(raw_words[start:end + 1]),
                    "probability": float(probabilities[index, start, end, label_id]),
                    "strongest_nearby": strongest_nearby,
                })
            profiles.append({"text": "".join(raw_words), "tokens": raw_words, "gold": gold})
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("cases", "profiles"), default="cases")
    parser.add_argument("--ablation", choices=("snsa", "hsr"), default="snsa")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    experiments = [
        (
            "Full SPSR-Net",
            "/root/shared-nvme/projects/SPSR-Net/_saved_models/2026-07-03-00_07_48_429272/"
            "model-epoch_15-batch_45015-f#f#test_97.98/fastnlp_model.pkl.tar",
            1,
            1,
        ),
        (
            "w/o SNSA",
            "/root/shared-nvme/projects/SPSR-Net/_saved_models/2026-07-04-20_43_24_622169/"
            "model-epoch_6-batch_18006-f#f#test_97.13/fastnlp_model.pkl.tar",
            0,
            1,
        ),
    ]
    if args.ablation == "hsr":
        experiments[1] = (
            "w/o HSR",
            "/root/shared-nvme/projects/SPSR-Net/_saved_models/2026-07-05-22_12_33_633670/"
            "model-epoch_8-batch_24008-f#f#test_95.87/fastnlp_model.pkl.tar",
            1,
            0,
        )
    if args.mode == "cases":
        payload = [
            case
            for name, checkpoint, use_sad, use_hsr in experiments
            for case in export_cases(checkpoint, use_sad, name, device, use_hsr)
        ]
    else:
        payload = {
            name: profile_gold_scores(checkpoint, use_sad, name, device, use_hsr)
            for name, checkpoint, use_sad, use_hsr in experiments
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
