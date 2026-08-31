import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/root/data/food_jsonline")
OUT = ROOT / "data" / "datasets" / "food"


def read_jsonlines(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def convert_record(record):
    entities = []
    for ent in record.get("entity_mentions", []):
        start = int(ent["start"])
        end = int(ent["end"])
        if 0 <= start < end <= len(record["tokens"]):
            entities.append({
                "type": ent["entity_type"],
                "start": start,
                "end": end,
            })

    return {
        "tokens": record["tokens"],
        "pos": ["NN"] * len(record["tokens"]),
        "entities": entities,
        "relations": [],
        "ltokens": [],
        "rtokens": [],
        "org_id": record.get("sent_id") or record.get("doc_id") or "",
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = []
    summary = {"source": str(SOURCE), "splits": {}}
    for split in ("train", "dev", "test"):
        records = read_jsonlines(SOURCE / f"{split}.jsonlines")
        converted = [convert_record(record) for record in records]
        with open(OUT / f"{split}_context.json", "w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False)
        split_labels = {}
        max_len = 0
        max_ent_len = 0
        ent_count = 0
        for doc in converted:
            max_len = max(max_len, len(doc["tokens"]))
            for ent in doc["entities"]:
                ent_count += 1
                split_labels[ent["type"]] = split_labels.get(ent["type"], 0) + 1
                max_ent_len = max(max_ent_len, ent["end"] - ent["start"])
        labels.extend(split_labels)
        summary["splits"][split] = {
            "documents": len(converted),
            "tokens": sum(len(doc["tokens"]) for doc in converted),
            "entities": ent_count,
            "labels": split_labels,
            "max_tokens": max_len,
            "max_entity_tokens": max_ent_len,
        }

    ordered_labels = []
    schema_path = SOURCE / "label_schema.json"
    if schema_path.exists():
        schema = json.load(open(schema_path, "r", encoding="utf-8"))
        ordered_labels.extend(schema.get("labels", []))
    for label in sorted(set(labels)):
        if label not in ordered_labels:
            ordered_labels.append(label)

    types = {
        "entities": {
            label: {
                "short": label,
                "verbose": label,
            }
            for label in ordered_labels
        },
        "relations": {},
    }
    with open(OUT / "food_types.json", "w", encoding="utf-8") as f:
        json.dump(types, f, ensure_ascii=False, indent=2)
    with open(OUT / "food_pos.json", "w", encoding="utf-8") as f:
        json.dump({"NN": 1000000}, f, ensure_ascii=False, indent=2)

    # The current experiment disables the word-vector branch, but JsonInputReader
    # still expects a glove-style file in order to initialize its vocabulary.
    with open(OUT / "glove.6B.2d.txt", "w", encoding="utf-8") as f:
        f.write("<unk> 0.0 0.0\n")

    summary["labels"] = ordered_labels
    with open(OUT / "prepare_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
