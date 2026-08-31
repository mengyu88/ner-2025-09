#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def bioes_labels(tokens, entities):
    labels = ["O"] * len(tokens)
    occupied = [False] * len(tokens)
    skipped = 0
    for ent in sorted(entities, key=lambda item: (item["start"], item["end"])):
        start = int(ent["start"])
        end = int(ent["end"])
        ent_type = ent.get("entity_type") or ent.get("type")
        if start < 0 or end > len(tokens) or start >= end:
            skipped += 1
            continue
        if any(occupied[start:end]):
            skipped += 1
            continue
        span_len = end - start
        if span_len == 1:
            labels[start] = f"S-{ent_type}"
        else:
            labels[start] = f"B-{ent_type}"
            for idx in range(start + 1, end - 1):
                labels[idx] = f"I-{ent_type}"
            labels[end - 1] = f"E-{ent_type}"
        for idx in range(start, end):
            occupied[idx] = True
    return labels, skipped


def read_jsonlines(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def convert_split(src_path, out_path):
    lengths = []
    entity_counts = Counter()
    skipped_entities = 0
    samples = 0
    with out_path.open("w", encoding="utf-8") as out:
        for sample in read_jsonlines(src_path):
            tokens = sample["tokens"]
            entities = sample.get("entity_mentions") or sample.get("entities") or []
            labels, skipped = bioes_labels(tokens, entities)
            skipped_entities += skipped
            for ent in entities:
                entity_counts[ent.get("entity_type") or ent.get("type")] += 1
            out.write(json.dumps({"text": tokens, "label": labels}, ensure_ascii=False) + "\n")
            lengths.append(len(tokens))
            samples += 1
    return {
        "samples": samples,
        "max_len": max(lengths) if lengths else 0,
        "avg_len": sum(lengths) / len(lengths) if lengths else 0,
        "entity_counts": dict(entity_counts),
        "skipped_overlapping_or_invalid_entities": skipped_entities,
    }


def build_lexicon_and_embedding(dataset_dir, vocab_out, embedding_out, max_words, max_ngram, dim):
    import hashlib

    counts = Counter()
    for split in ("train", "dev", "test"):
        with (dataset_dir / f"{split}.json").open("r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                text = sample["text"]
                for start in range(len(text)):
                    for width in range(1, max_ngram + 1):
                        end = start + width
                        if end > len(text):
                            break
                        word = "".join(text[start:end])
                        if word.strip():
                            counts[word] += 1

    words = sorted(counts, key=lambda item: (-counts[item], len(item), item))[:max_words]

    def vector_for(word):
        values = []
        state = hashlib.sha256(word.encode("utf-8")).digest()
        while len(values) < dim:
            for byte in state:
                values.append((byte / 255.0 - 0.5) * 0.2)
                if len(values) == dim:
                    break
            state = hashlib.sha256(state).digest()
        return values

    vocab_out.parent.mkdir(parents=True, exist_ok=True)
    embedding_out.parent.mkdir(parents=True, exist_ok=True)
    with vocab_out.open("w", encoding="utf-8") as f:
        for word in words:
            f.write(f"{word}\n")
    with embedding_out.open("w", encoding="utf-8") as f:
        f.write(f"{len(words)} {dim}\n")
        for word in words:
            vec = " ".join(f"{value:.6f}" for value in vector_for(word))
            f.write(f"{word} {vec}\n")
    return len(words)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="/root/data/food_jsonline")
    parser.add_argument("--output-dir", default="data/dataset/NER/food")
    parser.add_argument("--vocab-out", default="data/vocab/food_vocab.txt")
    parser.add_argument("--embedding-out", default="data/embedding/food_word_embedding.txt")
    parser.add_argument("--max-words", type=int, default=50000)
    parser.add_argument("--max-ngram", type=int, default=6)
    parser.add_argument("--dim", type=int, default=200)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = json.loads((source_dir / "label_schema.json").read_text(encoding="utf-8"))
    types = list(schema["labels"])
    labels = ["O"]
    for ent_type in types:
        labels.append(f"S-{ent_type}")
        labels.append(f"B-{ent_type}")
        labels.append(f"I-{ent_type}")
        labels.append(f"E-{ent_type}")
    (output_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    summary = {"labels": labels, "splits": {}}
    for split, filename in (("train", "train.jsonlines"), ("dev", "dev.jsonlines"), ("test", "test.jsonlines")):
        summary["splits"][split] = convert_split(source_dir / filename, output_dir / f"{split}.json")

    summary["lexicon_words"] = build_lexicon_and_embedding(
        output_dir,
        Path(args.vocab_out),
        Path(args.embedding_out),
        args.max_words,
        args.max_ngram,
        args.dim,
    )
    (output_dir / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
