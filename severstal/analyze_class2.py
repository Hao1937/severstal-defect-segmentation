"""CPU-only diagnosis of crop exposure and class-2 probabilities during training."""

import argparse
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from severstal.data import decode_rle, load_annotations, normalize
from severstal.models import make_model


def crop_exposure(labels, names, crop_width):
    classes = []
    for c in range(4):
        areas, retention, pixels = [], [], []
        for name in names:
            if not labels[name][c]:
                continue
            mask = decode_rle(labels[name][c])
            columns = mask.sum(0)
            prefix = np.r_[0, np.cumsum(columns)]
            window_areas = prefix[crop_width:] - prefix[:-crop_width]
            areas.append(int(mask.sum()))
            retention.append(float((window_areas > 0).mean()))
            pixels.append(float(window_areas.mean()))
        classes.append({
            "class_id": c + 1, "positive_images": len(areas),
            "full_mask_area_quantiles": np.quantile(areas, [0, .25, .5, .75, 1]).tolist(),
            "mean_positive_crop_probability_given_positive_image": float(np.mean(retention)),
            "expected_positive_crops_per_epoch": float(np.sum(retention)),
            "expected_positive_crop_fraction": float(np.sum(retention) / len(names)),
            "expected_positive_pixel_fraction": float(np.sum(pixels) / (len(names) * 256 * crop_width)),
        })
    return classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/E02-r01")
    parser.add_argument("--output", required=True)
    parser.add_argument("--negative-images", type=int, default=196)
    args = parser.parse_args()
    torch.set_num_threads(1)
    began = time.perf_counter()
    output, run = Path(args.output), Path(args.run)
    output.mkdir(parents=True, exist_ok=False)
    # Keep a stable checkpoint while the original run continues saving its own best.
    shutil.copyfile(run / "best.pt", output / "checkpoint.pt")
    shutil.copyfile(run / "history.jsonl", output / "history_snapshot.jsonl")
    state = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
    config = state["config"]
    split = json.loads((run / "split.json").read_text())
    labels = load_annotations(config["data_dir"])
    exposure = crop_exposure(labels, split["train"], config["crop_width"])
    (output / "crop_exposure.json").write_text(json.dumps(exposure, indent=2))
    print(json.dumps({"checkpoint_epoch": state["epoch"], "crop_exposure": exposure}), flush=True)

    positives = [n for n in split["val"] if labels[n][1]]
    negatives = [n for n in split["val"] if not labels[n][1]]
    sample = random.Random(42).sample(negatives, min(args.negative_images, len(negatives)))
    thresholds = [.1, .2, .3, .4, .5]
    model = make_model(config, pretrained=False).eval()
    model.load_state_dict(state["model"])
    rows = []
    with torch.inference_mode():
        for i, name in enumerate(positives + sample):
            with Image.open(Path(config["data_dir"]) / "train_images" / name) as source:
                tensor = normalize(source.convert("RGB")).unsqueeze(0)
            probability = model(tensor).sigmoid()[0, 1].numpy()
            truth = decode_rle(labels[name][1]).astype(bool)
            area = int(truth.sum())
            row = {"image": name, "true_area": area, "max_probability": float(probability.max()),
                   "mean_probability_on_truth": float(probability[truth].mean()) if area else None,
                   "max_probability_on_truth": float(probability[truth].max()) if area else None,
                   "thresholds": {}}
            for threshold in thresholds:
                mask = probability > threshold
                pred_area, overlap = int(mask.sum()), int((mask & truth).sum())
                denom = pred_area + area
                row["thresholds"][str(threshold)] = {"pred_area": pred_area, "intersection": overlap,
                                                     "dice": 2 * overlap / denom if denom else 1.0}
            rows.append(row)
            if (i + 1) % 49 == 0:
                print(json.dumps({"images_done": i + 1, "total": len(positives) + len(sample)}), flush=True)
    summaries = []
    for threshold in thresholds:
        p = [r["thresholds"][str(threshold)] for r in rows if r["true_area"]]
        n = [r["thresholds"][str(threshold)] for r in rows if not r["true_area"]]
        summaries.append({"threshold": threshold, "positive_images": len(p), "negative_images_sampled": len(n),
                          "detected_positive_images": sum(r["pred_area"] > 0 for r in p),
                          "positive_images_with_overlap": sum(r["intersection"] > 0 for r in p),
                          "nonempty_dice": float(np.mean([r["dice"] for r in p])),
                          "false_positive_images_in_sample": sum(r["pred_area"] > 0 for r in n)})
    gt_max = [r["max_probability_on_truth"] for r in rows if r["true_area"]]
    report = {"run": args.run, "checkpoint_epoch": state["epoch"], "training_commit": state["git_commit"],
              "checkout_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "inference": "cpu_float32_no_tta", "threshold_role": "diagnostic_only_not_selected",
              "positive_images": len(positives), "negative_population": len(negatives),
              "negative_sample_seed": 42, "negative_sample_size": len(sample),
              "gt_max_probability_quantiles": np.quantile(gt_max, [0, .25, .5, .75, 1]).tolist(),
              "thresholds": summaries, "seconds": time.perf_counter() - began}
    (output / "probability_rows.json").write_text(json.dumps(rows, indent=2))
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
