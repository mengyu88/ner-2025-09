import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "food"


def find_latest_run():
    logs_root = OUT / "logs" / "food_train"
    runs = [p for p in logs_root.iterdir() if p.is_dir()] if logs_root.exists() else []
    if not runs:
        raise SystemExit(f"No run directories found under {logs_root}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def read_eval(path):
    if not path.exists():
        return {}
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        delimiter = ";" if ";" in header and "," not in header else ","
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            epoch = int(row["epoch"])
            rows[epoch] = row
    return rows


def pct(value):
    value = float(value)
    if 0.0 <= value <= 1.0:
        value *= 100
    return f"{value:.2f}"


def main():
    run_dir = find_latest_run()
    valid = read_eval(run_dir / "eval_valid.csv")
    test = read_eval(run_dir / "eval_test.csv")
    epochs = sorted(set(valid) | set(test))
    out_path = OUT / "locate_label_food_epoch_entity_metrics.csv"
    OUT.mkdir(parents=True, exist_ok=True)

    fields = [
        "epoch",
        "step",
        "dev_entity_f1",
        "dev_entity_precision",
        "dev_entity_recall",
        "test_entity_f1",
        "test_entity_precision",
        "test_entity_recall",
        "remark",
    ]
    best_dev = -1.0
    rows = []
    for epoch in epochs:
        dev = valid.get(epoch, {})
        tst = test.get(epoch, {})
        dev_f1 = float(dev.get("ner_f1_micro", 0.0))
        remark = "best" if dev and dev_f1 > best_dev else ""
        if remark:
            best_dev = dev_f1
        rows.append({
            "epoch": epoch,
            "step": dev.get("global_iteration") or tst.get("global_iteration") or "",
            "dev_entity_f1": pct(dev.get("ner_f1_micro", 0.0)) if dev else "",
            "dev_entity_precision": pct(dev.get("ner_prec_micro", 0.0)) if dev else "",
            "dev_entity_recall": pct(dev.get("ner_rec_micro", 0.0)) if dev else "",
            "test_entity_f1": pct(tst.get("ner_f1_micro", 0.0)) if tst else "",
            "test_entity_precision": pct(tst.get("ner_prec_micro", 0.0)) if tst else "",
            "test_entity_recall": pct(tst.get("ner_rec_micro", 0.0)) if tst else "",
            "remark": remark,
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "run_dir": str(run_dir),
        "csv": str(out_path),
        "epochs": len(rows),
        "best_dev": max(rows, key=lambda r: float(r["dev_entity_f1"] or 0), default=None),
        "best_test": max(rows, key=lambda r: float(r["test_entity_f1"] or 0), default=None),
        "latest": rows[-1] if rows else None,
    }
    with open(OUT / "locate_label_food_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
