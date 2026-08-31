#!/usr/bin/env python3
"""Rank GENIA checkpoints created by the overnight no-PGD BHPC study."""

import csv
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT / "runs" / "genia_bhpc_replacement_20260818_overnight"
MARKER = RUN_ROOT / "baseline_no_aux" / "start.marker"
MODEL_ROOT = PROJECT / "_saved_models"


def main():
    if not MARKER.exists():
        raise SystemExit(f"Missing start marker: {MARKER}")
    since = MARKER.stat().st_mtime
    records = []
    for result_path in MODEL_ROOT.glob("*/fastnlp_evaluate_results.json"):
        if result_path.stat().st_mtime <= since:
            continue
        try:
            metrics = json.loads(result_path.read_text(encoding="utf-8"))
            if "f#f#dev" not in metrics or "f#f#test" not in metrics:
                continue
            records.append({
                "checkpoint": str(result_path.with_name("fastnlp_model.pkl.tar")),
                "checkpoint_dir": result_path.parent.name,
                "dev_f1": float(metrics["f#f#dev"]),
                "dev_precision": float(metrics.get("pre#f#dev", 0)),
                "dev_recall": float(metrics.get("rec#f#dev", 0)),
                "test_f1": float(metrics["f#f#test"]),
                "test_precision": float(metrics.get("pre#f#test", 0)),
                "test_recall": float(metrics.get("rec#f#test", 0)),
            })
        except (OSError, ValueError, TypeError):
            continue

    records.sort(key=lambda row: row["dev_f1"], reverse=True)
    output_json = RUN_ROOT / "ranked_genia_checkpoints.json"
    output_csv = RUN_ROOT / "ranked_genia_checkpoints.csv"
    payload = {
        "selection_rule": "Ranked by dev F1 only; test metrics are reported once per saved checkpoint.",
        "count": len(records),
        "best": records[0] if records else None,
        "records": records,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]) if records else ["checkpoint"])
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
