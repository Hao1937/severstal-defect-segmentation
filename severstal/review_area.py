"""Exact whole-mask area-filter review from saved per-image validation rows."""

import argparse
import json
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np


def summarize(rows, minimum_area):
    positives = [r for r in rows if r["true_area"] > 0]
    keep = lambda r: r["pred_area"] > 0 and r["pred_area"] >= minimum_area
    scores = [r["dice"] if keep(r) else float(r["true_area"] == 0) for r in rows]
    positive_scores = [r["dice"] if keep(r) else 0.0 for r in positives]
    return {"minimum_area": minimum_area, "dice": float(np.mean(scores)),
            "nonempty_dice": float(np.mean(positive_scores)),
            "detected_positive_images": sum(keep(r) for r in positives),
            "positive_images_with_overlap": sum(keep(r) and r["dice"] > 0 for r in positives),
            "false_negative_images": sum(not keep(r) for r in positives),
            "false_positive_images": sum(keep(r) and r["true_area"] == 0 for r in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/E02-r01")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run, output = Path(args.run), Path(args.output)
    metadata = (run / "best_metrics.json").read_text()
    rows_text = (run / "best_validation.jsonl").read_text()
    if metadata != (run / "best_metrics.json").read_text():
        raise RuntimeError("Best checkpoint changed during snapshot; run the review again")
    rows = [json.loads(line) for line in rows_text.splitlines()]
    expected = len(json.loads((run / "split.json").read_text())["val"]) * 4
    if len(rows) != expected:
        raise RuntimeError("Validation rows are incomplete; run the review after saving finishes")
    output.mkdir(parents=True, exist_ok=False)
    (output / "best_validation.jsonl").write_text(rows_text)
    (output / "best_metrics.json").write_text(metadata)
    shutil.copyfile(run / "history.jsonl", output / "history_snapshot.jsonl")
    results = []
    for c in range(1, 5):
        selected = [r for r in rows if r["class_id"] == c]
        results.append({"class_id": c,
                        "thresholds": [summarize(selected, n) for n in [0, 50, 100, 200, 400, 600, 1000, 2000]]})
    report = {"time": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "run": args.run, "epoch": json.loads(metadata)["epoch"],
              "pixel_threshold": .5, "validation_images": expected // 4,
              "filter": "whole_class_total_area", "purpose": "validation_diagnostic_not_independent_test",
              "classes": results}
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
