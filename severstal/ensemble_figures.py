"""Render per-class ensemble comparisons and preselected class-2 failure cases."""

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
from .evaluate_ensemble import average_probabilities
from .evaluate_gates import gate_masks
from .train import read_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    summary = json.loads((run / "summary.json").read_text())
    config = summary["config"]
    combinations = config["combinations"]
    x = np.arange(len(combinations))
    fig, axes = plt.subplots(2, 2, figsize=(max(13, 2.6 * len(combinations)), 9), constrained_layout=True)
    for class_id, ax in enumerate(axes.flat, 1):
        dice = [summary["combinations"][combo]["area"]["classes"][str(class_id)]["nonempty_dice"] for combo in combinations]
        bars = ax.bar(x, dice, color=["#4776b4" if len(combo) == 1 else "#d8882a" for combo in combinations])
        ax.set(xticks=x, xticklabels=combinations, ylim=(0, 1), title=f"Class {class_id} | common area filter", ylabel="Positive Dice")
        for bar, combo, score in zip(bars, combinations, dice):
            entry = summary["combinations"][combo]["area"]["classes"][str(class_id)]
            ax.text(bar.get_x() + bar.get_width() / 2, score + .015,
                    f"{score:.3f}\nFP {entry['false_positive_pairs']}\noverlap {entry['overlap_pairs']}/{entry['positive_pairs']}",
                    ha="center", fontsize=8)
    fig.suptitle("Equal-weight probability ensembles | fixed validation split and postprocessing")
    fig.savefig(run / "ensemble_comparison.png", dpi=120)
    plt.close(fig)
    data_dir = read_config(Path(config["members"]["A"]) / "config.yaml")["data_dir"]
    labels = load_annotations(data_dir)
    cases = run / "cases"
    cases.mkdir(exist_ok=True)
    selected = []
    for name in config.get("reference_cases", []):
        source = run / "reference_probabilities" / (Path(name).stem + ".npz")
        if not source.exists():
            continue
        with np.load(source) as stored:
            approximate = any(stored[member].dtype == np.float16 for member in config["members"])
            probabilities = {member: torch.from_numpy(stored[member].copy())[None] for member in config["members"]}
        image = np.asarray(Image.open(Path(data_dir) / "train_images" / name).convert("RGB"))
        target = decode_rle(labels[name][1]).astype(bool)
        fig, axes = plt.subplots(len(combinations) + 1, 1, figsize=(16, 2 * (len(combinations) + 1)), constrained_layout=True)
        precision_note = " | float16 cache: approximate boundaries/metrics" if approximate else ""
        fig.suptitle(f"Class 2 fixed reference | {name} | common area filter{precision_note}", fontsize=12)
        axes[0].imshow(image)
        axes[0].contour(target, levels=[.5], colors=["lime"], linewidths=.8)
        axes[0].set_title(f"Ground truth | area={int(target.sum())}", loc="left", fontsize=10)
        records = {}
        for ax, combo in zip(axes[1:], combinations):
            probability = average_probabilities(probabilities, combo)
            mask = gate_masks(probability, torch.ones((1, 4)), config)["area"][0, 1].numpy()
            overlay = np.zeros((*mask.shape, 4), dtype=float)
            overlay[mask] = [1, .2, .1, .45]
            ax.imshow(image)
            ax.imshow(overlay)
            ax.contour(target, levels=[.5], colors=["lime"], linewidths=.8)
            score = 2 * int((mask & target).sum()) / max(int(mask.sum() + target.sum()), 1)
            ax.set_title(f"{combo} | red=prediction, green=true boundary | area={int(mask.sum())} | Dice={score:.4f}", loc="left", fontsize=10)
            records[combo] = {"pred_area": int(mask.sum()), "dice": score}
        for ax in axes:
            ax.set_axis_off()
        destination = cases / (Path(name).stem + ".png")
        fig.savefig(destination, dpi=100)
        plt.close(fig)
        selected.append({"image": name, "reason": "fixed_A_failure_selected_before_B_training",
                         "figure": str(destination.relative_to(run)), "combinations": records,
                         "approximate_from_float16": approximate})
    (run / "selected_cases.json").write_text(json.dumps(selected, indent=2))
    print(json.dumps({"comparison": str(run / "ensemble_comparison.png"), "cases": len(selected)}))


if __name__ == "__main__":
    main()
