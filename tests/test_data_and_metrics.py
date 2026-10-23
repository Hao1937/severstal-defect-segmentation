import csv
import json

import numpy as np
import pytest
import torch
from PIL import Image

from severstal.data import decode_rle, encode_rle, load_annotations, make_split, SteelDataset
from severstal.metrics import DiceMeter


def test_rle_column_major_and_roundtrip():
    expected = np.array([[1, 0, 0], [1, 0, 1]], dtype=np.uint8)
    np.testing.assert_array_equal(decode_rle("1 2 6 1", 2, 3), expected)
    assert encode_rle(expected) == "1 2 6 1"
    assert encode_rle(np.zeros((2, 3), dtype=np.uint8)) == ""
    with pytest.raises(ValueError):
        decode_rle("6 2", 2, 3)


def test_official_dice_averages_pairs_including_empty():
    truth = torch.zeros(2, 4, 2, 2)
    prediction = truth.clone()
    prediction[0, 0, 0, 0] = 1  # An empty truth with one false-positive pixel scores zero.
    truth[1, 1, 0, 0] = 1       # A missed positive also scores zero.
    meter = DiceMeter()
    meter.update(prediction, truth, ["a", "b"])
    result = meter.summary()
    assert result["dice"] == 0.75
    assert result["nonempty_dice"] == 0
    assert result["classes"]["1"]["false_positive_pairs"] == 1
    assert result["classes"]["2"]["false_negative_pairs"] == 1


@pytest.mark.parametrize("legacy", [False, True])
def test_annotation_formats_and_split(tmp_path, legacy):
    (tmp_path / "train_images").mkdir()
    header = ["ImageId_ClassId", "EncodedPixels"] if legacy else ["ImageId", "ClassId", "EncodedPixels"]
    with (tmp_path / "train.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for i in range(40):
            name = f"{i:03}.jpg"
            Image.new("RGB", (4, 4)).save(tmp_path / "train_images" / name)
            if i % 5:
                c = i % 4 + 1
                writer.writerow([f"{name}_{c}", "1 2"] if legacy else [name, c, "1 2"])
    labels = load_annotations(tmp_path)
    assert labels["000.jpg"] == ["", "", "", ""]
    split = make_split(tmp_path, tmp_path / "split.json")
    assert not set(split["train"]) & set(split["val"])
    assert set(split["train"]) | set(split["val"]) == set(labels)
    assert split["counts"]["total"] == 40


def test_crop_recalculates_classification_target(tmp_path, monkeypatch):
    (tmp_path / "train_images").mkdir()
    Image.new("RGB", (4, 2)).save(tmp_path / "train_images" / "a.jpg")
    # Positive pixels are in the leftmost column; crop the rightmost half.
    dataset = SteelDataset(tmp_path, ["a.jpg"], {"a.jpg": ["1 2", "", "", ""]}, True, 2, 0)
    monkeypatch.setattr("severstal.data.random.randint", lambda *args: 2)
    monkeypatch.setattr("severstal.data.random.random", lambda: 1.0)
    image, mask, labels, _ = dataset[0]
    assert image.shape == (3, 2, 2)
    assert mask.sum() == 0
    assert labels.sum() == 0
