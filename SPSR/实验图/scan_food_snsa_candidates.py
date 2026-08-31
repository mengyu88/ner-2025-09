"""Find FoodReg test cases whose true span-representation similarities rise with SNSA.

The script is a selection aid only: it compares the real checkpoint features
for gold spans and prints candidates suitable for a qualitative heatmap.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).parent
PROJECT = Path("/root/shared-nvme/projects/SPSR-Net")
for entry in (SCRIPT_DIR, PROJECT, PROJECT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from eval_thresholds import build_model, load_data, move_to_device  # noqa: E402
from plot_food_similarity_three_layouts import CHECKPOINTS, MODEL_NAME, build_args, forward_features  # noqa: E402


def descriptors(batch):
    output = []
    for row, targets in enumerate(batch["ent_target"]):
        spans = sorted({(int(start), int(end), int(label)) for start, end, label in targets})
        # A six-row panel needs the target plus at least five gold neighbours.
        if len(spans) < 6:
            output.append([])
            continue
        candidates = []
        for target_start, target_end, target_label in spans:
            neighbours = sorted(
                (span for span in spans if span[:2] != (target_start, target_end)),
                key=lambda span: (
                    min(abs(span[0] - target_end), abs(target_start - span[1])),
                    span[0],
                    span[1],
                ),
            )[:5]
            candidates.append(((target_start, target_end, target_label), neighbours))
        output.append(candidates)
    return output


def evaluate(model_name, loader, matrix_segs, device):
    checkpoint, use_snsa, use_hsr = CHECKPOINTS[model_name]
    model = build_model(MODEL_NAME, matrix_segs, build_args(checkpoint, use_snsa, use_hsr)).to(device).eval()
    rows = {}
    for batch_index, batch in enumerate(loader):
        batch_candidates = descriptors(batch)
        if not any(batch_candidates):
            continue
        features, _ = forward_features(model, batch, device)
        for row, candidates in enumerate(batch_candidates):
            for target, neighbours in candidates:
                target_vector = F.normalize(features[row, target[0], target[1]].float(), dim=-1, eps=1e-8)
                spans = [target, *neighbours]
                vectors = F.normalize(
                    torch.stack([features[row, start, end] for start, end, _ in spans]).float(), dim=-1, eps=1e-8
                )
                rows[(batch_index, row, target)] = {
                    "text": "".join(batch["raw_words"][row]),
                    "spans": spans,
                    "phrases": ["".join(batch["raw_words"][row][start : end + 1]) for start, end, _ in spans],
                    "similarities": [float(value) for value in (vectors @ target_vector).tolist()],
                }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, matrix_segs = load_data(MODEL_NAME, "food", batch_size=4, num_workers=1)
    without = evaluate("w/o SNSA", loaders["test"], matrix_segs, device)
    full = evaluate("Full SPSR-Net", loaders["test"], matrix_segs, device)
    ranking = []
    for key, ablated in without.items():
        complete = full[key]
        deltas = [right - left for left, right in zip(ablated["similarities"][1:], complete["similarities"][1:])]
        # Strong candidates improve in every displayed relation, not merely on average.
        if min(deltas) > 0.0:
            ranking.append(
                {
                    "minimum_delta": min(deltas),
                    "mean_delta": sum(deltas) / len(deltas),
                    "text": complete["text"],
                    "spans": complete["spans"],
                    "phrases": complete["phrases"],
                    "without_snsa": ablated["similarities"],
                    "full": complete["similarities"],
                }
            )
    ranking.sort(key=lambda item: (item["mean_delta"], item["minimum_delta"]), reverse=True)
    print(json.dumps(ranking[:20], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
