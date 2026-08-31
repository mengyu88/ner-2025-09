#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def iter_texts(paths):
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                yield sample["text"]


def vector_for(word, dim):
    digest = hashlib.sha256(word.encode("utf-8")).digest()
    values = []
    state = digest
    while len(values) < dim:
        for byte in state:
            values.append((byte / 255.0 - 0.5) * 0.2)
            if len(values) == dim:
                break
        state = hashlib.sha256(state).digest()
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/dataset/NER/weibo")
    parser.add_argument("--vocab-out", default="data/vocab/tencent_vocab.txt")
    parser.add_argument("--embedding-out", default="data/embedding/word_embedding.txt")
    parser.add_argument("--max-words", type=int, default=8000)
    parser.add_argument("--max-ngram", type=int, default=4)
    parser.add_argument("--dim", type=int, default=200)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    paths = [data_dir / name for name in ("train.json", "dev.json", "test.json")]
    counts = {}
    for text in iter_texts(paths):
        for start in range(len(text)):
            for width in range(1, args.max_ngram + 1):
                end = start + width
                if end > len(text):
                    break
                word = "".join(text[start:end])
                if word.strip():
                    counts[word] = counts.get(word, 0) + 1

    words = sorted(counts, key=lambda item: (-counts[item], len(item), item))[: args.max_words]

    vocab_out = Path(args.vocab_out)
    embedding_out = Path(args.embedding_out)
    vocab_out.parent.mkdir(parents=True, exist_ok=True)
    embedding_out.parent.mkdir(parents=True, exist_ok=True)

    with vocab_out.open("w", encoding="utf-8") as f:
        for word in words:
            f.write(f"{word}\n")

    with embedding_out.open("w", encoding="utf-8") as f:
        f.write(f"{len(words)} {args.dim}\n")
        for word in words:
            vec = " ".join(f"{value:.6f}" for value in vector_for(word, args.dim))
            f.write(f"{word} {vec}\n")

    print(f"wrote {len(words)} words to {vocab_out}")
    print(f"wrote embeddings to {embedding_out}")


if __name__ == "__main__":
    main()
