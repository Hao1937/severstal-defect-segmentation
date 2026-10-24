"""Run frozen ABC inference against the images actually mounted by Kaggle."""

import argparse
import csv
import importlib.metadata
import json
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import encode_rle, normalize
from .evaluate_ensemble import average_probabilities
from .evaluate_gates import gate_masks
from .models import make_model


def image_names(data, names_file=None, limit=0):
    if names_file:
        names = json.loads(Path(names_file).read_text())
    else:
        names = sorted(p.name for p in data.glob("*.jpg"))
    if not names or len(set(names)) != len(names):
        raise ValueError("Image list must be nonempty and unique")
    return names[:limit] if limit else names


@torch.inference_mode()
def run(bundle, images, output, branches=("area", "soft"), names_file=None, limit=0):
    bundle, images, output = Path(bundle), Path(images), Path(output)
    if not set(branches) <= {"area", "soft"} or not branches:
        raise ValueError("Only the two predeclared branches are supported")
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU for this submission")
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((bundle / "manifest.json").read_text())
    names = image_names(images, names_file, limit)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    models = {}
    for name, member in config["members"].items():
        if name == "classifier" and "soft" not in branches:
            continue
        model = make_model(member["config"], pretrained=False)
        model.load_state_dict(torch.load(bundle / member["file"], map_location="cpu", weights_only=True))
        models[name] = model.cuda().eval()
    began = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    counts = {branch: np.zeros(4, dtype=np.int64) for branch in branches}
    with ExitStack() as stack:
        writers = {}
        for branch in branches:
            writer = csv.writer(stack.enter_context((output / f"submission_{branch}.csv").open("w", newline="")))
            writer.writerow(["ImageId_ClassId", "EncodedPixels"])
            writers[branch] = writer
        for index, name in enumerate(names):
            with Image.open(images / name) as image:
                if image.size != (1600, 256):
                    raise ValueError(f"Unexpected image size: {name} {image.size}")
                inputs = normalize(image.convert("RGB"))[None].cuda()
            probabilities = {}
            for member in "ABC":
                with torch.autocast("cuda", enabled=config["members"][member]["config"]["amp"]):
                    probabilities[member] = models[member](inputs).sigmoid()
            averaged = average_probabilities(probabilities, "ABC")
            q = torch.ones((1, 4), device="cuda")
            if "soft" in branches:
                with torch.autocast("cuda", enabled=config["members"]["classifier"]["config"]["amp"]):
                    q = models["classifier"](inputs).sigmoid().float()
            masks = gate_masks(averaged, q, config)
            for branch in branches:
                array = masks[branch][0].cpu().numpy()
                counts[branch] += array.reshape(4, -1).any(axis=1)
                for class_index in range(4):
                    writers[branch].writerow([f"{name}_{class_index + 1}", encode_rle(array[class_index])])
            if (index + 1) % 250 == 0:
                print(json.dumps({"images": index + 1, "seconds": time.perf_counter() - began}), flush=True)
    result = {
        "images": len(names), "rows_per_submission": len(names) * 4, "smoke_run": bool(limit or names_file),
        "branches": list(branches), "seconds": time.perf_counter() - began,
        "positive_predictions": {k: v.tolist() for k, v in counts.items()},
        "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20,
        "device": torch.cuda.get_device_name(), "manifest": config,
        "runtime_packages": {p: importlib.metadata.version(p) for p in config["packages"]},
    }
    (output / "inference_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ["images", "seconds", "positive_predictions"]}))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--branch", choices=["area", "soft", "both"], default="both")
    parser.add_argument("--names-file")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    branches = ("area", "soft") if args.branch == "both" else (args.branch,)
    run(args.bundle, args.images, args.output, branches, args.names_file, args.limit)


if __name__ == "__main__":
    main()
