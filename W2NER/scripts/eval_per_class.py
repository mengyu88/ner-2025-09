#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import Counter
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import torch
from torch.utils.data import DataLoader

import config as config_module
import data_loader
import utils
from model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/food.json')
    parser.add_argument('--checkpoint', default='outputs/food/model.pt')
    parser.add_argument('--model-name', default='/root/.cache/modelscope/hub/AI-ModelScope/bert-base-chinese')
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=24)
    args = parser.parse_args()
    cfg = config_module.Config(SimpleNamespace(config=args.config))
    cfg.bert_name = args.model_name
    cfg.batch_size = args.batch_size
    cfg.logger = utils.get_logger('food_per_class')
    datasets, original = data_loader.load_data_bert(cfg)
    test_loader = DataLoader(datasets[2], batch_size=args.batch_size, collate_fn=data_loader.collate_fn,
                             shuffle=False, num_workers=4, drop_last=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    labels = {value: key for key, value in cfg.vocab.label2id.items() if value > 1}
    counts = Counter()
    with torch.no_grad():
        for batch in test_loader:
            entity_text = batch[-1]
            tensors = [item.to(device) for item in batch[:-1]]
            bert_inputs, _, grid_mask2d, pieces2word, dist_inputs, sent_length = tensors
            output = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length).argmax(-1)
            _, _, _, decoded = utils.decode(output.cpu().numpy(), entity_text, sent_length.cpu().numpy())
            for gold_text, predicted in zip(entity_text, decoded):
                gold = {
                    (tuple(indices), label)
                    for indices, label in (utils.convert_text_to_index(item) for item in gold_text)
                }
                predicted = {(tuple(indices), label) for indices, label in predicted}
                for _, label in gold: counts[labels[label], 'gold'] += 1
                for _, label in predicted: counts[labels[label], 'predicted'] += 1
                for _, label in gold & predicted: counts[labels[label], 'tp'] += 1
    total = Counter(); rows = []
    for label in sorted(labels.values()):
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
        json.dump({'checkpoint': args.checkpoint, 'rows': rows}, file, ensure_ascii=False, indent=2)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
