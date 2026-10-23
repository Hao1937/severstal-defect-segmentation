"""Original-image splits, column-major RLE and paired image/mask transforms."""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def decode_rle(encoded, height=256, width=1600):
    mask = np.zeros(height * width, dtype=np.uint8)
    if encoded:
        runs = np.fromstring(str(encoded), sep=" ", dtype=np.int64)
        if len(runs) % 2:
            raise ValueError("RLE must contain start/length pairs")
        starts, lengths = runs[::2] - 1, runs[1::2]
        if np.any(starts < 0) or np.any(lengths < 0) or np.any(starts + lengths > mask.size):
            raise ValueError("RLE extends outside the image")
        for start, length in zip(starts, lengths):
            mask[start:start + length] = 1
    return mask.reshape((height, width), order="F")


def encode_rle(mask):
    pixels = np.asarray(mask, dtype=np.uint8).ravel(order="F")
    changes = np.flatnonzero(np.diff(np.concatenate(([0], pixels, [0])))) + 1
    changes[1::2] -= changes[::2]
    return " ".join(map(str, changes))


def load_annotations(data_dir):
    root = Path(data_dir)
    image_ids = sorted(p.name for p in (root / "train_images").glob("*.jpg"))
    if not image_ids:
        raise ValueError(f"No training images in {root}")
    labels = {name: ["", "", "", ""] for name in image_ids}
    with (root / "train.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if "ImageId_ClassId" in row:
                name, class_id = row["ImageId_ClassId"].rsplit("_", 1)
            else:
                name, class_id = row["ImageId"], row["ClassId"]
            index = int(class_id) - 1
            if name not in labels or index not in range(4):
                raise ValueError(f"Unknown image/class: {name}, {class_id}")
            encoded = row["EncodedPixels"].strip()
            if encoded:
                if labels[name][index]:
                    raise ValueError(f"Duplicate nonempty annotation: {name}, {class_id}")
                labels[name][index] = encoded
    return labels


def make_split(data_dir, output, seed=42, validation_fraction=0.2):
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    labels = load_annotations(data_dir)
    names = np.array(list(labels))
    y = np.array([[bool(v) for v in labels[name]] for name in names], dtype=np.uint8)
    stratify = np.column_stack([y, ~y.any(axis=1)])
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    train, val = next(splitter.split(names, stratify))
    result = {
        "seed": seed, "validation_fraction": validation_fraction,
        "method": "multilabel_stratified_original_images",
        "train": names[train].tolist(), "val": names[val].tolist(),
        "counts": {"total": len(names), "train": len(train), "val": len(val),
                   "positive_total": y.sum(axis=0).tolist(),
                   "positive_train": y[train].sum(axis=0).tolist(),
                   "positive_val": y[val].sum(axis=0).tolist()},
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2)
    return result


def normalize(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


class SteelDataset(Dataset):
    def __init__(self, data_dir, names, labels, training=False, crop_width=512, jitter=0.1):
        self.root = Path(data_dir)
        self.names = names
        self.labels = labels
        self.training = training
        self.crop_width = crop_width
        self.jitter = jitter

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        with Image.open(self.root / "train_images" / name) as source:
            image = np.asarray(source.convert("RGB")).copy()
        h, w = image.shape[:2]
        mask = np.stack([decode_rle(r, h, w) for r in self.labels[name]])
        if self.training:
            left = random.randint(0, max(0, w - self.crop_width))
            image = image[:, left:left + self.crop_width]
            mask = mask[:, :, left:left + self.crop_width]
            if random.random() < 0.5:
                image, mask = image[:, ::-1], mask[:, :, ::-1]
            if random.random() < 0.5:
                image, mask = image[::-1], mask[:, ::-1]
            if self.jitter:
                contrast = random.uniform(1 - self.jitter, 1 + self.jitter)
                brightness = random.uniform(-self.jitter, self.jitter) * 255
                image = np.clip(image.astype(np.float32) * contrast + brightness, 0, 255)
        # Derive classification targets after the paired crop.
        presence = mask.reshape(4, -1).any(axis=1).astype(np.float32)
        return normalize(image), torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)), torch.from_numpy(presence), name


class ClassificationDataset(SteelDataset):
    """Full-image labels come directly from RLE presence; no pixel masks are needed."""

    def __getitem__(self, index):
        name = self.names[index]
        with Image.open(self.root / "train_images" / name) as source:
            image = np.asarray(source.convert("RGB")).copy()
        if self.training:
            if self.crop_width != image.shape[1]:
                raise ValueError("ClassificationDataset requires full-image width to preserve labels")
            if random.random() < 0.5:
                image = image[:, ::-1]
            if random.random() < 0.5:
                image = image[::-1]
            if self.jitter:
                contrast = random.uniform(1 - self.jitter, 1 + self.jitter)
                brightness = random.uniform(-self.jitter, self.jitter) * 255
                image = np.clip(image.astype(np.float32) * contrast + brightness, 0, 255)
        presence = torch.tensor([bool(rle) for rle in self.labels[name]], dtype=torch.float32)
        return normalize(image), torch.empty(0), presence, name
