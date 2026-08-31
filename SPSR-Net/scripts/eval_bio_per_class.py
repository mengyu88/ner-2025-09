#!/usr/bin/env python3
"""Compute exact entity P/R/F1 for a three-column token/gold/pred file."""
import argparse
import json
from collections import Counter


def entities(tags):
    spans = []
    start = None
    entity_type = None
    for i, tag in enumerate(list(tags) + ['O']):
        if tag == 'O' or '-' not in tag:
            prefix, current_type = 'O', None
        else:
            prefix, current_type = tag.split('-', 1)
            if prefix == 'M':
                prefix = 'I'
        if start is not None and (prefix not in ('I', 'E') or current_type != entity_type):
            spans.append((start, i - 1, entity_type))
            start = None
            entity_type = None
        if prefix in ('B', 'S'):
            if prefix == 'S':
                spans.append((i, i, current_type))
            else:
                start, entity_type = i, current_type
        elif prefix in ('I', 'E') and start is None:
            start, entity_type = i, current_type
        if prefix == 'E' and start is not None:
            spans.append((start, i, current_type))
            start, entity_type = None, None
    return set(spans)


def read_sequences(path):
    gold, pred, seq_gold, seq_pred = [], [], [], []
    with open(path, encoding='utf-8') as file:
        for line in file:
            fields = line.strip().split()
            if not fields:
                if seq_gold:
                    gold.append(entities(seq_gold)); pred.append(entities(seq_pred))
                    seq_gold, seq_pred = [], []
                continue
            if len(fields) < 3:
                raise ValueError(f'Expected at least token/gold/pred columns, got: {line!r}')
            seq_gold.append(fields[1]); seq_pred.append(fields[2])
    if seq_gold:
        gold.append(entities(seq_gold)); pred.append(entities(seq_pred))
    return gold, pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    golds, preds = read_sequences(args.input)
    counts = Counter()
    for gold, pred in zip(golds, preds):
        for _, _, label in gold:
            counts[label, 'gold'] += 1
        for _, _, label in pred:
            counts[label, 'predicted'] += 1
        for _, _, label in gold & pred:
            counts[label, 'tp'] += 1
    labels = sorted({label for label, _ in counts})
    rows = []
    total = Counter()
    for label in labels:
        item = {key: counts[label, key] for key in ('tp', 'predicted', 'gold')}
        precision = 100 * item['tp'] / item['predicted'] if item['predicted'] else 0.0
        recall = 100 * item['tp'] / item['gold'] if item['gold'] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({'entity_type': label, 'precision': round(precision, 2), 'recall': round(recall, 2),
                     'f1': round(f1, 2), **item})
        total.update(item)
    precision = 100 * total['tp'] / total['predicted'] if total['predicted'] else 0.0
    recall = 100 * total['tp'] / total['gold'] if total['gold'] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    rows.append({'entity_type': 'Overall', 'precision': round(precision, 2), 'recall': round(recall, 2),
                 'f1': round(f1, 2), **dict(total)})
    with open(args.output, 'w', encoding='utf-8') as file:
        json.dump({'input': args.input, 'rows': rows}, file, ensure_ascii=False, indent=2)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
