"""Render reproducibly selected class-2 changes from a completed gate comparison."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from .data import decode_rle, load_annotations
from .evaluate_gates import gate_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", default="data/severstal")
    args = parser.parse_args()
    run = Path(args.run)
    summary = json.loads((run / "summary.json").read_text())
    changes = json.loads((run / "changed_cases.json").read_text())
    rows = {branch: [json.loads(line) for line in (run / f"{branch}_rows.jsonl").read_text().splitlines()]
            for branch in ("area", "hard", "dynamic")}
    classes = {r["image"]: r for r in map(json.loads, (run / "classifier_predictions.jsonl").read_text().splitlines())}
    selected, seen = [], set()

    def select(candidates, reason, score, limit=2):
        for row in sorted(candidates, key=lambda r: (-score(r), r["image"])):
            if row["image"] in seen:
                continue
            selected.append({**row, "reason": reason})
            seen.add(row["image"])
            if sum(item["reason"] == reason for item in selected) >= limit:
                break

    for branch in ("hard", "dynamic"):
        select([r for r in changes[branch] if r["class_id"] == 2 and r["change"] == "lost_overlap"],
               f"{branch}_lost_correct_localization", lambda r: r["area_dice"], limit=1)
    select([r for r in changes["dynamic"] if r["class_id"] == 2 and r["change"] == "removed_false_positive"],
           "dynamic_removed_false_positive", lambda r: r["area_pred_area"], limit=2)
    select([r for r in rows["dynamic"] if r["class_id"] == 2 and r["true_area"] > 0 and r["dice"] > 0
            and classes[r["image"]]["probability"][1] < 0.5],
           "dynamic_retained_despite_low_classifier_confidence", lambda r: r["dice"], limit=2)
    # If no gate vetoed a correct prediction, retain a high-quality reference example.
    select([r for r in rows["area"] if r["class_id"] == 2 and r["true_area"] > 0 and r["dice"] > 0],
           "area_reference", lambda r: r["dice"], limit=1)
    names = json.loads((run / "probability_names.json").read_text())
    indices = {name: index for index, name in enumerate(names)}
    probabilities = np.load(run / "segmentation_probability.npy", mmap_mode="r")
    labels = load_annotations(args.data)
    output = run / "cases"
    output.mkdir(exist_ok=True)
    for item in selected:
        name = item["image"]
        image = np.asarray(Image.open(Path(args.data) / "train_images" / name).convert("RGB"))
        target = decode_rle(labels[name][1]).astype(bool)
        p = torch.from_numpy(probabilities[indices[name]].copy())[None]
        q = torch.tensor([classes[name]["probability"]])
        masks = gate_masks(p, q, summary["config"])
        fig, axes = plt.subplots(6, 1, figsize=(16, 11.5), constrained_layout=True)
        fig.suptitle(f"Class 2 | {name} | classifier q={q[0, 1].item():.4f} | {item['reason']}", fontsize=12)
        axes[0].imshow(image)
        axes[0].set_title(f"Original | true area={int(target.sum())}", fontsize=10, loc="left")
        for ax, branch in zip(axes[1:], ("raw", "area", "hard", "soft", "dynamic")):
            mask = masks[branch][0, 1].numpy()
            overlay = np.zeros((*mask.shape, 4), dtype=float)
            overlay[mask] = [1, 0.2, 0.1, 0.45]
            ax.imshow(image)
            ax.imshow(overlay)
            if target.any():
                ax.contour(target, levels=[0.5], colors=["lime"], linewidths=0.8)
            denominator = int(mask.sum() + target.sum())
            dice = 2 * int((mask & target).sum()) / denominator if denominator else 1.0
            ax.set_title(f"{branch} | red=prediction, green=true boundary | area={int(mask.sum())} | Dice={dice:.4f}", fontsize=10, loc="left")
        for ax in axes:
            ax.set_axis_off()
        path = output / f"{Path(name).stem}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        item["figure"] = str(path.relative_to(run))
    (run / "selected_cases.json").write_text(json.dumps(selected, indent=2))
    print(json.dumps({"selected_cases": len(selected), "output": str(output)}))


if __name__ == "__main__":
    main()
