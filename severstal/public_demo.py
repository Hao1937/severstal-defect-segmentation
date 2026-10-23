"""Lightforever three-model migration: mean logits -> sigmoid -> inverse HFlip mean."""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .data import normalize, encode_rle


def save_overlay(image, masks, destination, label):
    palette = np.array([[255, 70, 70], [60, 220, 90], [60, 140, 255], [255, 210, 40]])
    original = np.asarray(image.convert("RGB"))
    overlay = original.astype(np.float32).copy()
    for mask, color in zip(masks, palette):
        overlay[mask] = overlay[mask] * 0.55 + color * 0.45
    panel = Image.new("RGB", (original.shape[1], original.shape[0] * 2 + 48), "white")
    panel.paste(Image.fromarray(original), (0, 24))
    panel.paste(Image.fromarray(overlay.astype(np.uint8)), (0, original.shape[0] + 48))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 5), label + " | original", fill="black")
    draw.text((8, original.shape[0] + 29), "prediction: class 1 red / 2 green / 3 blue / 4 yellow", fill="black")
    panel.save(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/severstal/test_images")
    parser.add_argument("--weights", default="weights/lightforever-converted")
    parser.add_argument("--output", default="runs/E01-r01")
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    files = sorted(Path(args.images).glob("*.jpg"))[:args.limit]
    if not files:
        raise ValueError("No images found")
    torch.set_num_threads(4)
    weights = ["unet_se_resnext50_32x4d.pth", "unet_mobilenet2.pth", "unet_resnet34.pth"]
    models = [torch.jit.load(str(Path(args.weights) / name), map_location="cuda").eval() for name in weights]
    minimum_area = [600, 600, 1000, 2000]
    rows, counts = [], np.zeros(4, dtype=int)
    began = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for index, path in enumerate(files):
            with Image.open(path) as source:
                image = source.convert("RGB")
                tensor = normalize(image).unsqueeze(0).cuda()
                tta = []
                for flip in (False, True):
                    x = tensor.flip(-1) if flip else tensor
                    logits = torch.stack([model(x) for model in models]).mean(0)
                    probability = logits.sigmoid()
                    tta.append(probability.flip(-1) if flip else probability)
                probability = torch.stack(tta).mean(0)[0].cpu().numpy()
                masks = probability > 0.5
                for c in range(4):
                    if masks[c].sum() < minimum_area[c]:
                        masks[c] = False
                    counts[c] += bool(masks[c].any())
                    rows.append({"ImageId_ClassId": f"{path.name}_{c + 1}", "EncodedPixels": encode_rle(masks[c])})
                np.savez_compressed(output / f"{path.stem}.npz", probability=probability.astype(np.float16))
                if masks.any() or index < 3:
                    save_overlay(image, masks, output / f"{path.stem}.png", path.name)
                print(json.dumps({"image": path.name, "area": masks.sum((1, 2)).tolist()}), flush=True)
    torch.cuda.synchronize()
    with (output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ImageId_ClassId", "EncodedPixels"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {"images": len(files), "seconds": time.perf_counter() - began,
               "positive_images_per_class": counts.tolist(), "device": torch.cuda.get_device_name(0),
               "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
               "weights": weights, "weights_directory": args.weights, "threshold": 0.5, "minimum_area": minimum_area,
               "tta": ["identity", "horizontal_flip"], "aggregation": "mean_logits_then_sigmoid_then_inverse_tta_mean",
               "batch_size": 1, "precision": "float32", "evaluation": "unlabeled_test_demo",
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "source": "https://www.kaggle.com/code/lightforever/severstal-mlcomp-catalyst-infer-0-90672"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
