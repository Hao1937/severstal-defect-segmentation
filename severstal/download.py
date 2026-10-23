"""Download the public mirror and original public model package on Legion."""

import argparse
import json
import subprocess
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path


SOURCES = {
    "data": ("https://www.kaggle.com/api/v1/datasets/download/krishar01/severstal-steel-defect-detection?datasetVersionNumber=1",
             "data/severstal.zip", "data/severstal"),
    "weights": ("https://www.kaggle.com/api/v1/datasets/download/lightforever/severstalmodels",
                "weights/lightforever.zip", "weights/lightforever"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["data", "weights", "all"], default="all", nargs="?")
    args = parser.parse_args()
    selected = SOURCES if args.kind == "all" else {args.kind: SOURCES[args.kind]}
    for kind, (url, archive_name, directory) in selected.items():
        archive, destination = Path(archive_name), Path(directory)
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists() or not zipfile.is_zipfile(archive):
            subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--continue-at", "-",
                            "--output", str(archive), url], check=True)
        if not destination.exists():
            destination.mkdir(parents=True)
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.namelist():
                    if not (destination / member).resolve().is_relative_to(destination.resolve()):
                        raise ValueError("Archive path leaves destination")
                bundle.extractall(destination)
        record = {"url": url, "archive_bytes": archive.stat().st_size, "directory": str(destination),
                  "recorded_at": datetime.now(timezone(timedelta(hours=8))).isoformat()}
        (archive.parent / f"{kind}_source.json").write_text(json.dumps(record, indent=2))
        print(json.dumps(record))


if __name__ == "__main__":
    main()
