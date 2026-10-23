"""Resave historical TorchScript with torch 1.4, then compare in current torch."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


MODELS = ["unet_se_resnext50_32x4d.pth", "unet_mobilenet2.pth", "unet_resnet34.pth"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "export", "verify"])
    parser.add_argument("--source", default="weights/lightforever")
    parser.add_argument("--destination", default="weights/lightforever-converted")
    parser.add_argument("--image", help="One real full-resolution test image for output comparison")
    args = parser.parse_args()
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        from PIL import Image
        from .data import normalize
        if not args.image:
            parser.error("prepare requires --image")
        with Image.open(args.image) as image:
            array = normalize(image.convert("RGB")).unsqueeze(0).numpy()
        np.save(destination / "comparison_input.npy", array)
        (destination / "input.json").write_text(json.dumps({"image": args.image, "shape": list(array.shape)}))
        return

    torch.set_num_threads(4)
    tensor = torch.from_numpy(np.load(destination / "comparison_input.npy"))
    report = {"mode": args.mode, "torch": torch.__version__, "device": "cpu", "models": []}
    with torch.no_grad():
        for name in MODELS:
            began = time.perf_counter()
            path = Path(args.source) / name if args.mode == "export" else destination / name
            model = torch.jit.load(str(path), map_location="cpu").eval()
            prediction = model(tensor).numpy()
            reference = destination / (name + ".reference.npy")
            row = {"model": name, "output_shape": list(prediction.shape)}
            if args.mode == "export":
                np.save(reference, prediction)
                torch.jit.save(model, str(destination / name))
            else:
                expected = np.load(reference)
                difference = np.abs(prediction - expected)
                row.update(max_absolute_error=float(difference.max()),
                           mean_absolute_error=float(difference.mean()),
                           zero_logit_mask_agreement=float(((prediction > 0) == (expected > 0)).mean()),
                           allclose=bool(np.allclose(prediction, expected, atol=1e-4, rtol=1e-4)))
            row["seconds"] = time.perf_counter() - began
            report["models"].append(row)
            print(json.dumps(row), flush=True)
            del model
    (destination / (args.mode + ".json")).write_text(json.dumps(report, indent=2))
    if args.mode == "verify" and not all(row["allclose"] for row in report["models"]):
        raise RuntimeError("Converted model outputs differ; inspect verify.json before using them")


if __name__ == "__main__":
    main()
