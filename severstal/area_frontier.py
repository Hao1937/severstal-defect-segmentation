"""Match dynamic-gate false positives with constant area filtering, without retraining."""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def area_curve(rows):
    """Enumerate every distinct whole-mask filtering result in ascending area."""
    groups = defaultdict(list)
    for row in rows:
        if row["pred_area"] > 0:
            groups[row["pred_area"]].append(row)
    positives = sum(row["true_area"] > 0 for row in rows)
    dice_sum = sum(row["dice"] for row in rows)
    positive_sum = sum(row["dice"] for row in rows if row["true_area"] > 0)
    fp = sum(row["true_area"] == 0 and row["pred_area"] > 0 for row in rows)
    fn = sum(row["true_area"] > 0 and row["pred_area"] == 0 for row in rows)
    overlap = sum(row["true_area"] > 0 and row["dice"] > 0 for row in rows)

    def snapshot(threshold):
        return {"minimum_area": threshold, "dice": dice_sum / len(rows),
                "nonempty_dice": positive_sum / positives if positives else None,
                "false_positive_pairs": fp, "false_negative_pairs": fn,
                "overlap_pairs": overlap, "positive_pairs": positives}

    result = [snapshot(0)]
    for area in sorted(groups):
        for row in groups[area]:
            if row["true_area"] > 0:
                dice_sum -= row["dice"]
                positive_sum -= row["dice"]
                fn += 1
                overlap -= int(row["dice"] > 0)
            else:
                dice_sum += 1
                fp -= 1
        result.append(snapshot(area + 1))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    summary = json.loads((run / "summary.json").read_text())
    rows = [json.loads(line) for line in (run / "raw_rows.jsonl").read_text().splitlines()]
    output = {"purpose": "validation_diagnostic_not_deployment_threshold_selection", "classes": {}}
    curves = {}
    for c in range(1, 5):
        curve = area_curve([r for r in rows if r["class_id"] == c])
        dynamic = summary["branches"]["dynamic"]["classes"][str(c)]
        # Tied areas can skip an exact match. Use the smallest threshold with no more FPs.
        matched = next(point for point in curve if point["false_positive_pairs"] <= dynamic["false_positive_pairs"])
        output["classes"][str(c)] = {
            "dynamic": dynamic, "fixed_area_at_no_more_false_positives": matched,
            "dynamic_minus_fixed_overlap": dynamic["overlap_pairs"] - matched["overlap_pairs"],
            "dynamic_minus_fixed_nonempty_dice": dynamic["nonempty_dice"] - matched["nonempty_dice"],
        }
        curves[str(c)] = curve
    (run / "matched_area_comparison.json").write_text(json.dumps(output, indent=2))
    (run / "fixed_area_frontiers.json").write_text(json.dumps(curves))
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
