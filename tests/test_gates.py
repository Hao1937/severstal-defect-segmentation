import torch

from severstal.evaluate_gates import gate_masks


CONFIG = {"pixel_threshold": 0.5, "classification_threshold": 0.5,
          "minimum_areas": [2, 2, 2, 2], "factor": 5}


def test_low_classifier_confidence_vetoes_hard_but_dynamic_can_keep_large_mask():
    p = torch.full((1, 4, 1, 10), 0.9)
    q = torch.tensor([[0.0, 0.25, 0.5, 1.0]])
    masks = gate_masks(p, q, CONFIG)
    assert masks["area"].flatten(2).sum(2).tolist() == [[10, 10, 10, 10]]
    assert masks["hard"].flatten(2).sum(2).tolist() == [[0, 0, 10, 10]]
    assert masks["soft"].flatten(2).sum(2).tolist() == [[0, 0, 0, 10]]
    assert masks["dynamic"].flatten(2).sum(2).tolist() == [[10, 10, 10, 10]]


def test_dynamic_interpolates_area_and_soft_can_shrink_a_retained_mask():
    p = torch.tensor([0.6, 0.9, 0.9, 0.9, 0.9, 0., 0., 0., 0., 0.]).repeat(1, 4, 1, 1)
    q = torch.tensor([[0., 0.5, 0.75, 1.]])
    masks = gate_masks(p, q, CONFIG)
    assert masks["dynamic"].flatten(2).sum(2).tolist() == [[0, 0, 5, 5]]
    assert masks["soft"].flatten(2).sum(2).tolist() == [[0, 0, 4, 5]]
    assert masks["fixed5"].sum() == 0
