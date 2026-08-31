import argparse
import collections
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastNLP import Vocabulary
from fastNLP.core import DataSetIter, RandomSampler, SequentialSampler, BucketSampler
from fastNLP.io.loader import ConllLoader

from fastNLP_module import StaticEmbedding
from utils import get_bigrams, norm_static_embedding
from V0.add_lattice import equip_chinese_ner_with_lexicon
from V0.models import Lattice_Transformer_SeqLabel


def read_jsonlines(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def spans_overlap(a, b):
    return max(a["start"], b["start"]) < min(a["end"], b["end"])


def flatten_entities(entities, strategy):
    valid = [
        e for e in entities
        if isinstance(e.get("start"), int)
        and isinstance(e.get("end"), int)
        and e["start"] < e["end"]
        and e.get("entity_type")
    ]
    if strategy == "longest":
        key = lambda e: (-(e["end"] - e["start"]), e["start"], e["end"], e["entity_type"])
    elif strategy == "shortest":
        key = lambda e: ((e["end"] - e["start"]), e["start"], e["end"], e["entity_type"])
    else:
        key = lambda e: (e["start"], -(e["end"] - e["start"]), e["end"], e["entity_type"])

    chosen = []
    for ent in sorted(valid, key=key):
        if all(not spans_overlap(ent, old) for old in chosen):
            chosen.append(ent)
    return sorted(chosen, key=lambda e: (e["start"], e["end"], e["entity_type"]))


def to_bmes(tokens, entities, strategy):
    tags = ["O"] * len(tokens)
    chosen = flatten_entities(entities, strategy)
    for ent in chosen:
        start, end, label = ent["start"], ent["end"], ent["entity_type"]
        if start < 0 or end > len(tokens) or start >= end:
            continue
        span_len = end - start
        if span_len == 1:
            tags[start] = f"S-{label}"
        else:
            tags[start] = f"B-{label}"
            for i in range(start + 1, end - 1):
                tags[i] = f"M-{label}"
            tags[end - 1] = f"E-{label}"
    return tags, chosen


def write_bmes(records, out_path, strategy):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dropped = 0
    chosen_gold = []
    all_gold = []
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            tokens = rec["tokens"]
            tags, chosen = to_bmes(tokens, rec.get("entity_mentions", []), strategy)
            dropped += len(rec.get("entity_mentions", [])) - len(chosen)
            chosen_gold.append({
                (ent["start"], ent["end"], ent["entity_type"])
                for ent in chosen
            })
            all_gold.append({
                (ent["start"], ent["end"], ent["entity_type"])
                for ent in rec.get("entity_mentions", [])
            })
            for token, tag in zip(tokens, tags):
                f.write(f"{token} {tag}\n")
            f.write("\n")
    return {"dropped": dropped, "flat_gold": chosen_gold, "all_gold": all_gold}


def build_lexicon(records, max_ngram_words, ngram_min_freq):
    lexicon = set()
    counter = collections.Counter()
    for rec in records:
        tokens = rec["tokens"]
        for ent in rec.get("entity_mentions", []):
            text = "".join(tokens[ent["start"]:ent["end"]])
            if len(text) >= 2:
                lexicon.add(text)
        n = len(tokens)
        for length in range(2, 7):
            for start in range(0, n - length + 1):
                word = "".join(tokens[start:start + length])
                if word.strip() == word:
                    counter[word] += 1

    for word, count in counter.most_common(max_ngram_words):
        if count >= ngram_min_freq and len(word) >= 2:
            lexicon.add(word)

    return sorted(lexicon, key=lambda x: (len(x), x))


def prepare_food_data(args):
    source_dir = Path(args.data_dir) / "food_jsonline"
    out_dir = ROOT / "data" / "food"
    split_records = {
        split: read_jsonlines(source_dir / f"{split}.jsonlines")
        for split in ("train", "dev", "test")
    }
    metadata = {}
    for split, records in split_records.items():
        metadata[split] = write_bmes(
            records,
            out_dir / f"{split}.char.bmes",
            args.flatten_strategy,
        )
    lexicon = build_lexicon(split_records["train"], args.max_ngram_words, args.ngram_min_freq)

    summary = {
        "source_dir": str(source_dir),
        "out_dir": str(out_dir),
        "flatten_strategy": args.flatten_strategy,
        "lexicon_size": len(lexicon),
        "splits": {
            split: {
                "sentences": len(records),
                "entities_all": sum(len(r.get("entity_mentions", [])) for r in records),
                "entities_dropped_for_flat_tags": metadata[split]["dropped"],
                "max_len": max(len(r["tokens"]) for r in records),
            }
            for split, records in split_records.items()
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "prepare_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_dir / "train_lexicon.txt", "w", encoding="utf-8") as f:
        for word in lexicon:
            f.write(word + "\n")
    return out_dir, metadata, lexicon, summary


def load_food_eval_metadata(source_dir, strategy):
    """Read the original FOOD annotations without rewriting the training files.

    The saved Flat-Lattice checkpoint was trained on ``data/food/*.char.bmes``.
    During evaluation we must keep those files unchanged, while scoring predictions
    against the original (possibly nested) JSONL annotations.
    """
    source_dir = Path(source_dir)
    split_records = {
        split: read_jsonlines(source_dir / f"{split}.jsonlines")
        for split in ("train", "dev", "test")
    }
    metadata = {}
    for split, records in split_records.items():
        all_gold = []
        flat_gold = []
        for record in records:
            valid = {
                (ent["start"], ent["end"], ent["entity_type"])
                for ent in record.get("entity_mentions", [])
                if isinstance(ent.get("start"), int)
                and isinstance(ent.get("end"), int)
                and ent["start"] < ent["end"]
                and ent.get("entity_type")
            }
            chosen = {
                (ent["start"], ent["end"], ent["entity_type"])
                for ent in flatten_entities(record.get("entity_mentions", []), strategy)
            }
            all_gold.append(valid)
            flat_gold.append(chosen)
        metadata[split] = {"all_gold": all_gold, "flat_gold": flat_gold}
    return split_records, metadata


def load_food_ner(path):
    loader = ConllLoader(["chars", "target"])
    datasets = {}
    for split in ("train", "dev", "test"):
        bundle = loader.load(str(path / f"{split}.char.bmes"))
        datasets[split] = bundle.datasets["train"]

    for dataset in datasets.values():
        dataset.apply_field(get_bigrams, field_name="chars", new_field_name="bigrams")
        dataset.add_seq_len("chars")

    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary(padding=None, unknown=None)
    char_vocab.from_dataset(
        datasets["train"],
        field_name="chars",
        no_create_entry_dataset=[datasets["dev"], datasets["test"]],
    )
    bigram_vocab.from_dataset(
        datasets["train"],
        field_name="bigrams",
        no_create_entry_dataset=[datasets["dev"], datasets["test"]],
    )
    label_vocab.from_dataset(datasets["train"], field_name="target")

    vocabs = {"char": char_vocab, "bigram": bigram_vocab, "label": label_vocab}
    return datasets, vocabs, {}


def split_tag(tag):
    if tag == "O" or tag == "":
        return "O", ""
    if "-" not in tag:
        return tag, ""
    prefix, label = tag.split("-", 1)
    return prefix, label


def tag_ids_to_spans(tag_ids, seq_len, label_vocab):
    tags = [label_vocab.to_word(int(tag_ids[i])) for i in range(int(seq_len))]
    spans = set()
    i = 0
    while i < len(tags):
        prefix, label = split_tag(tags[i])
        if prefix == "O" or not label:
            i += 1
            continue
        if prefix == "S":
            spans.add((i, i + 1, label))
            i += 1
            continue
        if prefix == "B":
            start = i
            i += 1
            while i < len(tags):
                mid_prefix, mid_label = split_tag(tags[i])
                if mid_prefix == "M" and mid_label == label:
                    i += 1
                    continue
                if mid_prefix == "E" and mid_label == label:
                    i += 1
                break
            if i > start + 1:
                spans.add((start, i, label))
            else:
                spans.add((start, start + 1, label))
            continue
        spans.add((i, i + 1, label))
        i += 1
    return spans


def prf(tp, fp, fn):
    precision = tp / (tp + fp + 1e-13)
    recall = tp / (tp + fn + 1e-13)
    f1 = 2 * precision * recall / (precision + recall + 1e-13)
    return {
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def move_to_device(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def evaluate(model, dataset, all_gold, label_vocab, batch_size, device):
    model.eval()
    iterator = DataSetIter(dataset, batch_size=batch_size, sampler=SequentialSampler())
    all_tp = all_fp = all_fn = 0
    flat_tp = flat_fp = flat_fn = 0
    offset = 0
    with torch.no_grad():
        for batch_x, batch_y in iterator:
            batch_size_now = int(batch_x["seq_len"].size(0))
            batch_x_dev = move_to_device(batch_x, device)
            output = model(**batch_x_dev)
            pred = output["pred"].detach().cpu()
            target = batch_x["target"]
            seq_len = batch_x["seq_len"]
            for row in range(batch_size_now):
                pred_spans = tag_ids_to_spans(pred[row], seq_len[row], label_vocab)
                flat_gold = tag_ids_to_spans(target[row], seq_len[row], label_vocab)
                full_gold = all_gold[offset + row]

                all_tp += len(pred_spans & full_gold)
                all_fp += len(pred_spans - full_gold)
                all_fn += len(full_gold - pred_spans)

                flat_tp += len(pred_spans & flat_gold)
                flat_fp += len(pred_spans - flat_gold)
                flat_fn += len(flat_gold - pred_spans)
            offset += batch_size_now
    all_scores = prf(all_tp, all_fp, all_fn)
    flat_scores = prf(flat_tp, flat_fp, flat_fn)
    return all_scores, flat_scores


def per_class_scores(model, dataset, gold_by_sentence, label_vocab, batch_size, device):
    """Exact-span P/R/F1, with one prediction counted under its predicted type."""
    model.eval()
    counts = collections.defaultdict(lambda: {"tp": 0, "pred": 0, "gold": 0})
    iterator = DataSetIter(dataset, batch_size=batch_size, sampler=SequentialSampler())
    offset = 0
    with torch.no_grad():
        for batch_x, _ in iterator:
            batch_size_now = int(batch_x["seq_len"].size(0))
            output = model(**move_to_device(batch_x, device))
            pred = output["pred"].detach().cpu()
            seq_len = batch_x["seq_len"]
            for row in range(batch_size_now):
                pred_spans = tag_ids_to_spans(pred[row], seq_len[row], label_vocab)
                gold_spans = gold_by_sentence[offset + row]
                for _, _, label in pred_spans:
                    counts[label]["pred"] += 1
                for _, _, label in gold_spans:
                    counts[label]["gold"] += 1
                for _, _, label in pred_spans & gold_spans:
                    counts[label]["tp"] += 1
            offset += batch_size_now

    result = {}
    labels = sorted(counts)
    total_tp = total_pred = total_gold = 0
    for label in labels:
        value = counts[label]
        tp, pred_count, gold_count = value["tp"], value["pred"], value["gold"]
        precision = 100 * tp / pred_count if pred_count else 0.0
        recall = 100 * tp / gold_count if gold_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[label] = {
            "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "pred": pred_count, "gold": gold_count,
        }
        total_tp += tp
        total_pred += pred_count
        total_gold += gold_count
    precision = 100 * total_tp / total_pred if total_pred else 0.0
    recall = 100 * total_tp / total_gold if total_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result["overall"] = {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": total_tp, "pred": total_pred, "gold": total_gold,
    }
    return result


def init_model_weights(model, init):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if (
                "embedding" not in name
                and "pos" not in name
                and "pe" not in name
                and "bias" not in name
                and "crf" not in name
                and param.dim() > 1
            ):
                if init == "uniform":
                    nn.init.xavier_uniform_(param)
                elif init == "norm":
                    nn.init.xavier_normal_(param)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(args, embeddings, vocabs, max_seq_len, device):
    dropout = collections.defaultdict(float)
    dropout["embed"] = args.embed_dropout
    dropout["gaz"] = args.gaz_dropout
    dropout["output"] = args.output_dropout
    dropout["pre"] = args.pre_dropout
    dropout["post"] = args.post_dropout
    dropout["ff"] = args.ff_dropout
    dropout["ff_2"] = args.ff_dropout_2
    dropout["attn"] = args.attn_dropout

    model = Lattice_Transformer_SeqLabel(
        embeddings["lattice"],
        embeddings["bigram"],
        hidden_size=args.hidden,
        label_size=len(vocabs["label"]),
        num_heads=args.head,
        num_layers=args.layer,
        use_abs_pos=False,
        use_rel_pos=True,
        learnable_position=False,
        add_position=False,
        layer_preprocess_sequence="",
        layer_postprocess_sequence="an",
        ff_size=args.ff,
        scaled=False,
        dropout=dropout,
        use_bigram=True,
        mode=collections.defaultdict(bool),
        dvc=device,
        vocabs=vocabs,
        rel_pos_shared=True,
        max_seq_len=max_seq_len,
        k_proj=False,
        q_proj=True,
        v_proj=True,
        r_proj=True,
        self_supervised=False,
        attn_ff=False,
        pos_norm=False,
        ff_activate="relu",
        rel_pos_init=1,
        abs_pos_fusion_func="nonlinear_add",
        embed_dropout_pos="0",
        four_pos_shared=True,
        four_pos_fusion="ff_two",
        four_pos_fusion_shared=True,
        use_pytorch_dropout=0,
    )
    init_model_weights(model, args.init)
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/root/data")
    parser.add_argument("--output_dir", default=str(ROOT / "outputs" / "food"))
    parser.add_argument("--eval_checkpoint", default="", help="Evaluate this saved model and exit.")
    parser.add_argument(
        "--eval_source_dir", default="",
        help="Directory containing train/dev/test.jsonlines; required with --eval_checkpoint.",
    )
    parser.add_argument(
        "--per_class_output", default="",
        help="Path for exact-span per-class JSON produced in evaluation mode.",
    )
    parser.add_argument("--flatten_strategy", default="shortest", choices=["shortest", "longest", "first"])
    parser.add_argument("--max_ngram_words", type=int, default=5000)
    parser.add_argument("--ngram_min_freq", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--early_stop", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_lr_rate", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=11741)
    parser.add_argument("--lattice_dim", type=int, default=100)
    parser.add_argument("--bigram_dim", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--ff", type=int, default=384)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--head", type=int, default=8)
    parser.add_argument("--embed_dropout", type=float, default=0.5)
    parser.add_argument("--gaz_dropout", type=float, default=0.5)
    parser.add_argument("--output_dropout", type=float, default=0.3)
    parser.add_argument("--pre_dropout", type=float, default=0.5)
    parser.add_argument("--post_dropout", type=float, default=0.3)
    parser.add_argument("--ff_dropout", type=float, default=0.3)
    parser.add_argument("--ff_dropout_2", type=float, default=0.3)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--init", default="uniform", choices=["uniform", "norm"])
    parser.add_argument("--clip_grad_norm", type=float, default=5.0)
    parser.add_argument("--num_buckets", type=int, default=30)
    parser.add_argument("--limit_train", type=int, default=0)
    parser.add_argument("--limit_dev", type=int, default=0)
    parser.add_argument("--limit_test", type=int, default=0)
    parser.add_argument("--print_every", type=int, default=100)
    args = parser.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "flat_lattice_food_epoch_entity_metrics.csv"
    best_path = output_dir / "model.pt"
    summary_path = output_dir / "flat_lattice_food_summary.json"

    if args.eval_checkpoint:
        if not args.eval_source_dir:
            parser.error("--eval_source_dir is required with --eval_checkpoint")
        split_records, metadata = load_food_eval_metadata(
            args.eval_source_dir, args.flatten_strategy,
        )
        data_path = ROOT / "data" / "food"
        lexicon = build_lexicon(
            split_records["train"], args.max_ngram_words, args.ngram_min_freq,
        )
        data_summary = {
            "source_dir": str(args.eval_source_dir),
            "out_dir": str(data_path),
            "flatten_strategy": args.flatten_strategy,
            "mode": "evaluation_without_rewriting_bmes",
            "splits": {
                split: {
                    "sentences": len(records),
                    "entities_all": sum(len(r.get("entity_mentions", [])) for r in records),
                    "entities_dropped_for_flat_tags": sum(
                        len(metadata[split]["all_gold"][i]) - len(metadata[split]["flat_gold"][i])
                        for i in range(len(records))
                    ),
                }
                for split, records in split_records.items()
            },
        }
    else:
        data_path, metadata, lexicon, data_summary = prepare_food_data(args)
    datasets, vocabs, embeddings = load_food_ner(data_path)

    datasets, vocabs, embeddings = equip_chinese_ner_with_lexicon(
        datasets,
        vocabs,
        embeddings,
        lexicon,
        word_embedding_path=None,
        only_lexicon_in_train=False,
        word_char_mix_embedding_path=None,
        number_normalized=False,
        lattice_min_freq=1,
        only_train_min_freq=False,
        _refresh=True,
        _cache_fp=str(ROOT / "cache" / "food_lattice_train_lexicon"),
    )

    for split, limit in (("train", args.limit_train), ("dev", args.limit_dev), ("test", args.limit_test)):
        if limit and limit > 0:
            datasets[split] = datasets[split][:limit]
            metadata[split]["all_gold"] = metadata[split]["all_gold"][:limit]
            metadata[split]["flat_gold"] = metadata[split]["flat_gold"][:limit]

    embeddings["bigram"] = StaticEmbedding(
        vocabs["bigram"],
        model_dir_or_name=None,
        embedding_dim=args.bigram_dim,
        word_dropout=0.01,
    )
    embeddings["lattice"] = StaticEmbedding(
        vocabs["lattice"],
        model_dir_or_name=None,
        embedding_dim=args.lattice_dim,
        word_dropout=0.01,
    )
    for embedding in embeddings.values():
        norm_static_embedding(embedding, 1)

    for dataset in datasets.values():
        dataset.apply(lambda ins: ins["seq_len"] + ins["lex_num"], new_field_name="seq_lex_len")
        dataset.set_input("lattice", "bigrams", "seq_len", "target", "lex_num", "pos_s", "pos_e")
        dataset.set_target("target", "seq_len")

    max_seq_len = max(max(dataset["seq_len"]) for dataset in datasets.values())
    max_seq_lex_len = max(max(dataset["seq_lex_len"]) for dataset in datasets.values())
    model = make_model(args, embeddings, vocabs, max_seq_len, device)

    if args.eval_checkpoint:
        checkpoint = torch.load(args.eval_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        all_result = per_class_scores(
            model, datasets["test"], metadata["test"]["all_gold"], vocabs["label"],
            args.eval_batch_size, device,
        )
        flat_result = per_class_scores(
            model, datasets["test"], metadata["test"]["flat_gold"], vocabs["label"],
            args.eval_batch_size, device,
        )
        result = {
            "model": "Flat-Lattice-Transformer",
            "checkpoint": str(args.eval_checkpoint),
            "scoring": "exact span and exact entity type",
            "all_gold": all_result,
            "flat_gold": flat_result,
            "note": "flat_gold follows the shortest non-overlap BMES conversion used in training; all_gold is the original FOOD annotation.",
        }
        per_class_path = Path(args.per_class_output) if args.per_class_output else output_dir / "flat_lattice_per_class.json"
        per_class_path.parent.mkdir(parents=True, exist_ok=True)
        with open(per_class_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return

    bigram_embedding_param = list(model.bigram_embed.parameters())
    lattice_embedding_param = list(model.lattice_embed.parameters())
    embedding_param = bigram_embedding_param + lattice_embedding_param
    embedding_param_ids = set(map(id, embedding_param))
    non_embedding_param = [
        param for param in model.parameters()
        if id(param) not in embedding_param_ids
    ]
    param_groups = [
        {"params": non_embedding_param},
        {"params": embedding_param, "lr": args.lr * args.embed_lr_rate},
    ]
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(param_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    train_bucket_count = max(1, min(args.num_buckets, len(datasets["train"])))
    train_sampler = BucketSampler(
        num_buckets=train_bucket_count,
        batch_size=args.batch_size,
        seq_len_field_name="seq_lex_len",
    )

    best_dev_f1 = -math.inf
    best_record = None
    bad_epochs = 0
    total_step = 0
    label_list = [vocabs["label"].to_word(i) for i in range(len(vocabs["label"]))]
    run_config = {
        "args": vars(args),
        "device": str(device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "labels": label_list,
        "data_summary": data_summary,
        "vocab_sizes": {
            "char": len(vocabs["char"]),
            "bigram": len(vocabs["bigram"]),
            "lattice": len(vocabs["lattice"]),
            "label": len(vocabs["label"]),
        },
        "max_seq_len": int(max_seq_len),
        "max_seq_lex_len": int(max_seq_lex_len),
        "effective_train_size": len(datasets["train"]),
        "effective_dev_size": len(datasets["dev"]),
        "effective_test_size": len(datasets["test"]),
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "epoch",
        "step",
        "train_loss",
        "dev_entity_f1",
        "dev_entity_precision",
        "dev_entity_recall",
        "test_entity_f1",
        "test_entity_precision",
        "test_entity_recall",
        "dev_flat_f1",
        "dev_flat_precision",
        "dev_flat_recall",
        "test_flat_f1",
        "test_flat_precision",
        "test_flat_recall",
        "lr",
        "remark",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    print(json.dumps(run_config, ensure_ascii=False), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        iterator = DataSetIter(datasets["train"], batch_size=args.batch_size, sampler=train_sampler)
        total_loss = 0.0
        total_batches = 0
        for batch_x, _ in iterator:
            batch_x = move_to_device(batch_x, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch_x)
            loss = output["loss"]
            loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            total_step += 1
            total_loss += float(loss.detach().cpu())
            total_batches += 1
            if args.print_every > 0 and total_batches % args.print_every == 0:
                print(
                    f"epoch={epoch} batch={total_batches}/{len(iterator)} "
                    f"loss={total_loss / total_batches:.4f}",
                    flush=True,
                )

        train_loss = total_loss / max(total_batches, 1)
        dev_scores, dev_flat_scores = evaluate(
            model,
            datasets["dev"],
            metadata["dev"]["all_gold"],
            vocabs["label"],
            args.eval_batch_size,
            device,
        )
        test_scores, test_flat_scores = evaluate(
            model,
            datasets["test"],
            metadata["test"]["all_gold"],
            vocabs["label"],
            args.eval_batch_size,
            device,
        )
        improved = dev_scores["f1"] > best_dev_f1
        remark = "best" if improved else ""
        if improved:
            best_dev_f1 = dev_scores["f1"]
            bad_epochs = 0
            best_record = {
                "epoch": epoch,
                "step": total_step,
                "dev": dev_scores,
                "test": test_scores,
                "dev_flat": dev_flat_scores,
                "test_flat": test_flat_scores,
            }
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "run_config": run_config,
                    "best_record": best_record,
                },
                best_path,
            )
        else:
            bad_epochs += 1

        row = {
            "epoch": epoch,
            "step": total_step,
            "train_loss": f"{train_loss:.6f}",
            "dev_entity_f1": f"{dev_scores['f1']:.2f}",
            "dev_entity_precision": f"{dev_scores['precision']:.2f}",
            "dev_entity_recall": f"{dev_scores['recall']:.2f}",
            "test_entity_f1": f"{test_scores['f1']:.2f}",
            "test_entity_precision": f"{test_scores['precision']:.2f}",
            "test_entity_recall": f"{test_scores['recall']:.2f}",
            "dev_flat_f1": f"{dev_flat_scores['f1']:.2f}",
            "dev_flat_precision": f"{dev_flat_scores['precision']:.2f}",
            "dev_flat_recall": f"{dev_flat_scores['recall']:.2f}",
            "test_flat_f1": f"{test_flat_scores['f1']:.2f}",
            "test_flat_precision": f"{test_flat_scores['precision']:.2f}",
            "test_flat_recall": f"{test_flat_scores['recall']:.2f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.8f}",
            "remark": remark,
        }
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        summary = {
            "best": best_record,
            "latest": row,
            "csv": str(csv_path),
            "checkpoint": str(best_path),
            "config": run_config,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(
            "epoch={epoch} loss={loss:.4f} dev_f1={dev:.2f} test_f1={test:.2f} "
            "dev_flat_f1={dev_flat:.2f} test_flat_f1={test_flat:.2f} {remark}".format(
                epoch=epoch,
                loss=train_loss,
                dev=dev_scores["f1"],
                test=test_scores["f1"],
                dev_flat=dev_flat_scores["f1"],
                test_flat=test_flat_scores["f1"],
                remark=remark,
            ),
            flush=True,
        )

        if bad_epochs >= args.early_stop:
            print(f"early stop at epoch {epoch}", flush=True)
            break

    print(json.dumps({"best": best_record, "csv": str(csv_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
