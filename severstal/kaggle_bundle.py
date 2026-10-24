"""Export selected inference weights and project code for an offline Kaggle run."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    members = {"A": "E02-r01", "B": "E02-r03", "C": "E02-r04", "classifier": "E02-r02"}
    manifest = {
        "export_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "members": {}, "branches": ["area", "soft"],
        "pixel_threshold": 0.5, "minimum_areas": [100, 50, 200, 400],
        "classification_threshold": 0.5, "factor": 5,
        "packages": {p: importlib.metadata.version(p) for p in
                     ["torch", "torchvision", "segmentation-models-pytorch", "timm", "numpy", "Pillow"]},
    }
    split = None
    for name, run in members.items():
        directory = Path("runs") / run
        current_split = json.loads((directory / "split.json").read_text())
        if split is not None and current_split != split:
            raise ValueError("Member splits differ")
        split = current_split
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        destination = root / f"{name}.pt"
        torch.save(checkpoint["model"], destination)
        manifest["members"][name] = {
            "file": destination.name, "run": run, "epoch": checkpoint["epoch"],
            "training_commit": checkpoint["git_commit"], "config": checkpoint["config"],
            "sha256": hashlib.file_digest(destination.open("rb"), "sha256").hexdigest(),
        }
        del checkpoint
    shutil.copytree(Path(__file__).parent, root / "severstal", ignore=shutil.ignore_patterns("__pycache__"))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"output": str(root), "members": {k: v["epoch"] for k, v in manifest["members"].items()},
                      "weight_bytes": sum(p.stat().st_size for p in root.glob("*.pt"))}))


if __name__ == "__main__":
    main()
