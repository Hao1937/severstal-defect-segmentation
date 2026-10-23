"""Compare aligned member probabilities and their equal-weight combinations."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SteelDataset, load_annotations
from .evaluate_gates import gate_masks
from .metrics import DiceMeter
from .models import make_model
from .train import read_config


def average_probabilities(probabilities, members):
    """Average sigmoid probabilities in float32, before thresholding or filtering."""
    if not members or len(set(members)) != len(members):
        raise ValueError("A combination must contain distinct members")
    values = [probabilities[name].float() for name in members]
    if any(value.shape != values[0].shape for value in values):
        raise ValueError("Member prediction shapes differ")
    return torch.stack(values).mean(0)


def compare_rows(rows, reference):
    """Use image-class identities; report gains and losses separately."""
    indexed = {(r["image"], r["class_id"]): r for r in reference}
    if len(indexed) != len(reference) or len(rows) != len(reference):
        raise ValueError("Comparison has duplicate or missing rows")
    keys = [(r["image"], r["class_id"]) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != set(indexed):
        raise ValueError("Comparison identities differ")
    counts = {str(c): {"gained_overlap": 0, "lost_overlap": 0,
                       "added_false_positives": 0, "removed_false_positives": 0}
              for c in range(1, 5)}
    changes = []
    for row in rows:
        base = indexed[(row["image"], row["class_id"])]
        if row["true_area"] != base["true_area"]:
            raise ValueError("Comparison targets differ")
        kind = None
        if row["true_area"] > 0:
            if row["dice"] > 0 and base["dice"] == 0:
                kind = "gained_overlap"
            elif row["dice"] == 0 and base["dice"] > 0:
                kind = "lost_overlap"
        elif row["pred_area"] > 0 and base["pred_area"] == 0:
            kind = "added_false_positives"
        elif row["pred_area"] == 0 and base["pred_area"] > 0:
            kind = "removed_false_positives"
        if kind:
            counts[str(row["class_id"])][kind] += 1
            changes.append({**row, "change": kind, "reference_dice": base["dice"],
                            "reference_pred_area": base["pred_area"]})
    return counts, changes


def summarize(meter):
    result = meter.summary()
    for c in range(1, 5):
        positive = [r for r in meter.rows if r["class_id"] == c and r["true_area"] > 0]
        result["classes"][str(c)].update(
            positive_pairs=len(positive), overlap_pairs=sum(r["dice"] > 0 for r in positive))
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Smoke check only; default uses all validation images")
    args = parser.parse_args()
    config = read_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    models, model_configs, provenance, original_rows = {}, {}, {}, {}
    split = None
    for name, location in config["members"].items():
        run = Path(location)
        member_split = json.loads((run / "split.json").read_text())
        if split is not None and split != member_split:
            raise ValueError(f"Different split for member {name}")
        split = member_split
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        member_config = checkpoint["config"]
        if member_config["task"] != "segmentation":
            raise ValueError("Ensemble members must be segmenters")
        if model_configs and member_config["data_dir"] != next(iter(model_configs.values()))["data_dir"]:
            raise ValueError("Member data directories differ")
        model = make_model(member_config, pretrained=False)
        model.load_state_dict(checkpoint["model"])
        models[name] = model.cuda().eval()
        model_configs[name] = member_config
        provenance[name] = {"run": str(run), "epoch": checkpoint["epoch"],
                            "training_commit": checkpoint["git_commit"],
                            "architecture": member_config["architecture"], "encoder": member_config["encoder"],
                            "parameters": sum(p.numel() for p in model.parameters())}
        original_rows[name] = {(r["image"], r["class_id"]): r for r in
                              map(json.loads, (run / "best_validation.jsonl").read_text().splitlines())}
        del checkpoint
    if "A" not in models or "raw" not in config["branches"] or "area" not in config["branches"]:
        raise ValueError("A, raw and area are required comparison references")
    for combination in config["combinations"]:
        if not combination or len(set(combination)) != len(combination) or not set(combination) <= set(models):
            raise ValueError(f"Invalid combination: {combination}")
    names = split["val"][:args.limit] if args.limit else split["val"]
    labels = load_annotations(next(iter(model_configs.values()))["data_dir"])
    needs_classifier = bool(set(config["branches"]) & {"hard", "soft", "dynamic"})
    classification = None
    classifier_provenance = None
    if needs_classifier:
        cls_run = Path(config["classification_run"])
        if json.loads((cls_run / "split.json").read_text()) != split:
            raise ValueError("Classifier split differs")
        cls_list = list(map(json.loads, (cls_run / "best_validation.jsonl").read_text().splitlines()))
        classification = {r["image"]: r for r in cls_list}
        if len(classification) != len(cls_list) or set(classification) != set(split["val"]):
            raise ValueError("Classifier predictions do not cover validation exactly once")
        for name in names:
            if classification[name]["target"] != [float(bool(rle)) for rle in labels[name]]:
                raise ValueError(f"Classifier target differs: {name}")
        classifier_provenance = {"run": str(cls_run), "epoch": json.loads((cls_run / "best_metrics.json").read_text())["epoch"]}
    dataset = SteelDataset(next(iter(model_configs.values()))["data_dir"], names, labels)
    loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, shuffle=False)
    meters = {combo: {branch: DiceMeter() for branch in config["branches"]} for combo in config["combinations"]}
    forward_seconds = {member: 0.0 for member in models}
    cuda_start, cuda_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    saved = output / "reference_probabilities"
    saved.mkdir()
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    for index, (images, masks, _, batch_names) in enumerate(loader):
        images, masks = images.cuda(non_blocking=True), masks.cuda(non_blocking=True)
        probabilities = {}
        for member, model in models.items():
            cuda_start.record()
            with torch.autocast("cuda", enabled=model_configs[member]["amp"]):
                probability = model(images).sigmoid()
            cuda_end.record()
            cuda_end.synchronize()
            forward_seconds[member] += cuda_start.elapsed_time(cuda_end) / 1000
            probabilities[member] = probability.float()
        q = (torch.tensor([classification[name]["probability"] for name in batch_names], device="cuda")
             if classification is not None else torch.ones((len(batch_names), 4), device="cuda"))
        for combo, branches in meters.items():
            averaged = average_probabilities(probabilities, combo)
            predictions = gate_masks(averaged, q, config)
            for branch, meter in branches.items():
                meter.update(predictions[branch], masks, batch_names)
        if batch_names[0] in config.get("reference_cases", []):
            np.savez_compressed(saved / (Path(batch_names[0]).stem + ".npz"),
                                **{member: p[0].cpu().numpy().astype(np.float16) for member, p in probabilities.items()})
        if (index + 1) % 250 == 0:
            print(json.dumps({"images": index + 1, "seconds": time.perf_counter() - began}), flush=True)
    total_seconds = time.perf_counter() - began
    summaries, changes, reproduction = {}, {}, {}
    for combo, branches in meters.items():
        summaries[combo], changes[combo] = {}, {}
        for branch, meter in branches.items():
            entry = summarize(meter)
            entry["vs_A"], changes[combo][branch] = compare_rows(meter.rows, meters["A"][branch].rows)
            entry["vs_members"] = {member: compare_rows(meter.rows, meters[member][branch].rows)[0] for member in combo}
            summaries[combo][branch] = entry
            with (output / f"{combo}_{branch}_rows.jsonl").open("w") as stream:
                for row in meter.rows:
                    stream.write(json.dumps(row) + "\n")
        if len(combo) == 1:
            raw = branches["raw"].rows
            prior = original_rows[combo]
            reproduction[combo] = {
                "changed_area_pairs": sum(r["pred_area"] != prior[(r["image"], r["class_id"])]["pred_area"] for r in raw),
                "max_absolute_dice_delta": max(abs(r["dice"] - prior[(r["image"], r["class_id"])]["dice"]) for r in raw),
                "changed_overlap_pairs": sum((r["true_area"] > 0 and r["dice"] > 0) !=
                                             (prior[(r["image"], r["class_id"])]["true_area"] > 0 and prior[(r["image"], r["class_id"])]["dice"] > 0) for r in raw)}
    result = {"config": config, "members": provenance, "classifier": classifier_provenance,
              "evaluation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "images": len(names), "smoke_run": bool(args.limit), "seconds": total_seconds,
              "forward_seconds": forward_seconds,
              "timing_scope": "CUDA forward+sigmoid only per member; total includes I/O and all comparisons, excludes model loading and saved classifier inference",
              "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20,
              "raw_reproduction": reproduction, "combinations": summaries}
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    (output / "changed_cases.json").write_text(json.dumps(changes, indent=2))
    positives = {combo: {branch: [r for r in meter.rows if r["class_id"] == 2 and r["true_area"] > 0]
                         for branch, meter in branches.items()} for combo, branches in meters.items()}
    (output / "class2_positive_comparison.json").write_text(json.dumps(positives, indent=2))
    (output / "split.json").write_text(json.dumps(split))
    print(json.dumps({"event": "complete", "output": str(output), "images": len(names), "seconds": total_seconds}), flush=True)


if __name__ == "__main__":
    main()
