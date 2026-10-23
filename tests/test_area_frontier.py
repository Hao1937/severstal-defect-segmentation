import pytest

from severstal.area_frontier import area_curve


def test_equal_area_groups_are_removed_together_and_empty_pairs_stay_correct():
    rows = [
        {"pred_area": 3, "true_area": 5, "dice": 0.5},
        {"pred_area": 3, "true_area": 0, "dice": 0.0},
        {"pred_area": 10, "true_area": 8, "dice": 0.8},
        {"pred_area": 0, "true_area": 0, "dice": 1.0},
        {"pred_area": 0, "true_area": 2, "dice": 0.0},
    ]
    curve = area_curve(rows)
    assert [row["minimum_area"] for row in curve] == [0, 4, 11]
    assert curve[1]["false_positive_pairs"] == 0
    assert curve[1]["false_negative_pairs"] == 2
    assert curve[1]["overlap_pairs"] == 1
    assert curve[1]["nonempty_dice"] == pytest.approx(0.8 / 3)
    assert curve[-1]["dice"] == pytest.approx(2 / 5)
