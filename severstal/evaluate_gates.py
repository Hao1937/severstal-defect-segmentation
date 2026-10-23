"""Compare frozen gates on one segmentation checkpoint and saved classifier outputs."""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SteelDataset, load_annotations
from .metrics import DiceMeter
from .models import make_model
from .train import read_config


def gate_masks(probability, classification, config):
    """Inputs are B,C,H,W and B,C. Areas filter whole image-class masks."""
    raw = probability > config["pixel_threshold"]
    area = torch.as_tensor(config["minimum_areas"], device=probability.device)[None, :]
    q = classification[..., None, None]

    def filtered(mask, limit):
        keep = mask.flatten(2).sum(2) >= limit
        return mask & keep[..., None, None]

    return {
        "raw": raw,
        "area": filtered(raw, area),
        "fixed5": filtered(raw, config["factor"] * area),
        "hard": filtered(raw & (q >= config["classification_threshold"]), area),
        # Multiplication uses float32, preserving the original half-precision sigmoid values.
        "soft": filtered(probability.float() * q > config["pixel_threshold"], area),
        "dynamic": filtered(raw, area * (config["factor"] - (config["factor"] - 1) * classification)),
    }


def summarize(meter, baseline_rows, class_rows):
    summary = meter.summary()
    changed = []
    for c in range(1, 5):
        rows = [r for r in meter.rows if r["class_id"] == c]
        positive = [r for r in rows if r["true_area"] > 0]
        detail = summary["classes"][str(c)]
        detail["positive_pairs"] = len(positive)
        detail["overlap_pairs"] = sum(r["dice"] > 0 for r in positive)
        detail["lost_overlap_vs_area"] = 0
        detail["removed_false_positives_vs_area"] = 0
        for row in rows:
            base = baseline_rows[(row["image"], c)]
            lost = row["true_area"] > 0 and base["dice"] > 0 and row["dice"] == 0
            removed_fp = row["true_area"] == 0 and base["pred_area"] > 0 and row["pred_area"] == 0
            detail["lost_overlap_vs_area"] += int(lost)
            detail["removed_false_positives_vs_area"] += int(removed_fp)
            if lost or removed_fp:
                changed.append({**row, "classification_probability": class_rows[row["image"]]["probability"][c - 1],
                                "area_pred_area": base["pred_area"], "area_dice": base["dice"],
                                "change": "lost_overlap" if lost else "removed_false_positive"})
        # Fixed descriptive bins, not thresholds selected to improve a gate.
        detail["positive_area_bins"] = {}
        for label, lower, upper in [("1-999", 1, 1000), ("1000-4999", 1000, 5000), ("5000+", 5000, float("inf"))]:
            group = [r for r in positive if lower <= r["true_area"] < upper]
            detail["positive_area_bins"][label] = {
                "count": len(group), "overlap": sum(r["dice"] > 0 for r in group),
                "nonempty_dice": float(np.mean([r["dice"] for r in group])) if group else None,
            }
    return summary, changed


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gating.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    seg_run, cls_run = Path(config["segmentation_run"]), Path(config["classification_run"])
    seg_split = json.loads((seg_run / "split.json").read_text())
    cls_split = json.loads((cls_run / "split.json").read_text())
    if seg_split != cls_split:
        raise ValueError("Segmentation and classification must share the same split")
    names = seg_split["val"]
    cls_list = [json.loads(line) for line in (cls_run / "best_validation.jsonl").read_text().splitlines()]
    class_rows = {row["image"]: row for row in cls_list}
    if set(class_rows) != set(names) or len(cls_list) != len(names):
        raise ValueError("Classifier predictions do not cover validation exactly once")
    checkpoint = torch.load(seg_run / "best.pt", map_location="cpu", weights_only=False)
    seg_config = checkpoint["config"]
    labels = load_annotations(seg_config["data_dir"])
    for name in names:
        if class_rows[name]["target"] != [float(bool(rle)) for rle in labels[name]]:
            raise ValueError(f"Classifier target mismatch: {name}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    for source, target in [(cls_run / "best_validation.jsonl", "classifier_predictions.jsonl"),
                           (cls_run / "best_metrics.json", "classifier_best_metrics.json"),
                           (seg_run / "best_metrics.json", "segmentation_best_metrics.json")]:
        shutil.copyfile(source, output / target)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    model = make_model(seg_config, pretrained=False).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    dataset = SteelDataset(seg_config["data_dir"], names, labels)
    loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, shuffle=False)
    meters = {name: DiceMeter() for name in config["branches"]}
    cache = None
    if config["cache_probabilities"]:
        shape = (len(names), 4, 256, 1600)
        size = int(np.prod(shape)) * 2
        if shutil.disk_usage(output).free < size + 2 * 2**30:
            raise RuntimeError("Insufficient disk for probability cache plus 2 GiB headroom")
        cache = np.lib.format.open_memmap(output / "segmentation_probability.npy", mode="w+", dtype="float16", shape=shape)
        (output / "probability_names.json").write_text(json.dumps(names))
    began = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for index, (images, masks, _, batch_names) in enumerate(loader):
        images, masks = images.cuda(non_blocking=True), masks.cuda(non_blocking=True)
        with torch.autocast("cuda", enabled=seg_config["amp"]):
            probability = model(images).sigmoid()
        q = torch.tensor([class_rows[name]["probability"] for name in batch_names], device="cuda", dtype=torch.float32)
        for branch, predictions in gate_masks(probability, q, config).items():
            if branch in meters:
                meters[branch].update(predictions, masks, batch_names)
        if cache is not None:
            cache[index] = probability[0].cpu().numpy()
        if (index + 1) % 250 == 0:
            print(json.dumps({"images": index + 1, "seconds": time.perf_counter() - began}), flush=True)
    if cache is not None:
        cache.flush()
    baseline = {(r["image"], r["class_id"]): r for r in meters["area"].rows}
    summary, changes = {}, {}
    for branch, meter in meters.items():
        summary[branch], changes[branch] = summarize(meter, baseline, class_rows)
        with (output / f"{branch}_rows.jsonl").open("w") as stream:
            for row in meter.rows:
                stream.write(json.dumps(row) + "\n")
    original = {(r["image"], r["class_id"]): r for r in map(json.loads, (seg_run / "best_validation.jsonl").read_text().splitlines())}
    diffs = [{"image": row["image"], "class_id": row["class_id"],
              "area_delta": row["pred_area"] - original[(row["image"], row["class_id"])]["pred_area"],
              "dice_delta": row["dice"] - original[(row["image"], row["class_id"])]["dice"]}
             for row in meters["raw"].rows]
    reproduction = {"changed_area_pairs": sum(r["area_delta"] != 0 for r in diffs),
                    "max_absolute_dice_delta": max(abs(r["dice_delta"]) for r in diffs)}
    result = {"config": config, "segmentation_epoch": checkpoint["epoch"],
              "classification_epoch": json.loads((cls_run / "best_metrics.json").read_text())["epoch"],
              "segmentation_training_commit": checkpoint["git_commit"],
              "evaluation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "images": len(names), "seconds": time.perf_counter() - began,
              "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20,
              "raw_reproduction": reproduction, "branches": summary}
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    (output / "changed_cases.json").write_text(json.dumps(changes, indent=2))
    (output / "raw_reproduction.json").write_text(json.dumps([r for r in diffs if r["area_delta"] or r["dice_delta"]], indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
