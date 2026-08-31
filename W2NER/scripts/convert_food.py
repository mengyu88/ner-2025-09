import argparse
import json
from pathlib import Path


def read_items(input_dir: Path, split: str):
    json_path = input_dir / f"{split}.json"
    jsonl_path = input_dir / f"{split}.jsonlines"
    if json_path.exists():
        with json_path.open(encoding="utf-8") as f:
            return json.load(f), "json"
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()], "jsonlines"
    raise FileNotFoundError(f"Missing {split}.json or {split}.jsonlines under {input_dir}")


def convert_entity(entity, source_format):
    if source_format == "json":
        start = entity["start"]
        end = entity["end"]
        entity_type = entity["type"]
    else:
        start = entity["start"]
        end = entity["end"]
        entity_type = entity["entity_type"]

    return {
        "index": list(range(start, end)),
        "type": entity_type,
    }


def convert_item(item, source_format):
    entities_key = "entities" if source_format == "json" else "entity_mentions"
    return {
        "sentence": item["tokens"],
        "ner": [convert_entity(entity, source_format) for entity in item.get(entities_key, [])],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/root/data/food_json"))
    parser.add_argument("--output", type=Path, default=Path("data/food"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        items, source_format = read_items(args.input, split)
        converted = [convert_item(item, source_format) for item in items]
        with (args.output / f"{split}.json").open("w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False)
        entity_count = sum(len(item["ner"]) for item in converted)
        print(f"{split}: {len(converted)} sentences, {entity_count} entities from {source_format}")


if __name__ == "__main__":
    main()
