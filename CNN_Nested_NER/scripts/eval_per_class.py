#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')

import numpy as np
import torch
from fastNLP import SortedSampler, prepare_torch_dataloader

from data.ner_pipe import SpanNerPipe
from data.padder import Torch3DMatrixPadder
from model.metrics_utils import is_clashed
from model.model import CNNNer


def densify(x):
    return x.todense().astype(np.float32)


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def decode(scores, length, threshold):
    scores = scores.sigmoid()
    scores = (scores + scores.transpose(1, 2)) / 2
    output = []
    for sample, size in zip(scores, length.tolist()):
        sample = sample[:size, :size]
        values, labels = sample.max(dim=-1)
        starts, ends = torch.triu_indices(size, size)
        keep = values[starts, ends] >= threshold
        starts, ends = starts[keep], ends[keep]
        candidates = sorted(
            [(float(values[s, e]), int(labels[s, e]), int(s), int(e)) for s, e in zip(starts, ends)],
            reverse=True,
        )
        selected = []
        for _, label, start, end in candidates:
            item = (label, start, end)
            if all(not is_clashed(item, existed, allow_nested=True) for existed in selected):
                selected.append(item)
        output.append({(start, end, label) for label, start, end in selected})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--cnn-depth', type=int, default=2)
    parser.add_argument('--cnn-dim', type=int, default=120)
    parser.add_argument('--logit-drop', type=float, default=0.1)
    parser.add_argument('--biaffine-size', type=int, default=200)
    parser.add_argument('--n-head', type=int, default=5)
    args = parser.parse_args()

    pipe = SpanNerPipe(model_name=args.model_name)
    bundle = pipe.process_from_file(args.data_dir)
    bundle.apply_field(densify, field_name='matrix', new_field_name='matrix')
    labels = [label for label, _ in sorted(bundle.label2idx.items(), key=lambda item: item[1])]
    test = bundle.get_dataset('test')
    test.set_pad('matrix', pad_fn=Torch3DMatrixPadder(
        pad_val=test.collator.input_fields['matrix']['pad_val'], num_class=pipe.matrix_segs['ent'],
        batch_size=args.batch_size))
    loader = prepare_torch_dataloader(test, batch_size=args.batch_size, num_workers=args.num_workers,
                                      sampler=SortedSampler(test, 'input_ids'), pin_memory=True, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNNer(args.model_name, pipe.matrix_segs['ent'], cnn_dim=args.cnn_dim,
                   biaffine_size=args.biaffine_size, size_embed_dim=25, logit_drop=args.logit_drop,
                   kernel_size=3, n_head=args.n_head, cnn_depth=args.cnn_depth).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model.eval()
    counts = Counter()
    with torch.no_grad():
        for batch in loader:
            batch_on_device = to_device(batch, device)
            result = model(input_ids=batch_on_device['input_ids'], bpe_len=batch_on_device['bpe_len'],
                           indexes=batch_on_device['indexes'], matrix=batch_on_device['matrix'])
            for gold, predicted in zip(batch['ent_target'], decode(result['scores'].cpu(), batch['word_len'], args.threshold)):
                gold = set(map(tuple, gold))
                for _, _, label in gold: counts[labels[label], 'gold'] += 1
                for _, _, label in predicted: counts[labels[label], 'predicted'] += 1
                for _, _, label in gold & predicted: counts[labels[label], 'tp'] += 1
    total = Counter(); rows = []
    for label in labels:
        item = {key: counts[label, key] for key in ('tp', 'predicted', 'gold')}
        precision = 100 * item['tp'] / item['predicted'] if item['predicted'] else 0
        recall = 100 * item['tp'] / item['gold'] if item['gold'] else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        rows.append({'entity_type': label, 'precision': round(precision, 2), 'recall': round(recall, 2),
                     'f1': round(f1, 2), **item}); total.update(item)
    precision = 100 * total['tp'] / total['predicted'] if total['predicted'] else 0
    recall = 100 * total['tp'] / total['gold'] if total['gold'] else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    rows.append({'entity_type': 'Overall', 'precision': round(precision, 2), 'recall': round(recall, 2),
                 'f1': round(f1, 2), **dict(total)})
    with open(args.output, 'w', encoding='utf-8') as file:
        json.dump({'checkpoint': args.checkpoint, 'threshold': args.threshold, 'rows': rows}, file, ensure_ascii=False, indent=2)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
