import pytest
from PIL import Image

from severstal.data import ClassificationDataset


def test_full_image_classification_preserves_multilabel_target(tmp_path):
    (tmp_path / "train_images").mkdir()
    Image.new("RGB", (8, 4)).save(tmp_path / "train_images" / "sample.jpg")
    labels = {"sample.jpg": ["1 2", "", "9 1", ""]}
    dataset = ClassificationDataset(tmp_path, ["sample.jpg"], labels, True, 8, 0)
    image, masks, target, name = dataset[0]
    assert image.shape == (3, 4, 8)
    assert masks.numel() == 0
    assert target.tolist() == [1, 0, 1, 0]
    assert name == "sample.jpg"
    labels["sample.jpg"] = ["", "", "", ""]
    assert dataset[0][2].sum() == 0


def test_classification_rejects_crops_with_full_image_labels(tmp_path):
    (tmp_path / "train_images").mkdir()
    Image.new("RGB", (8, 4)).save(tmp_path / "train_images" / "sample.jpg")
    dataset = ClassificationDataset(tmp_path, ["sample.jpg"], {"sample.jpg": ["1 1", "", "", ""]}, True, 4, 0)
    with pytest.raises(ValueError, match="full-image"):
        dataset[0]
