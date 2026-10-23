import pytest
import torch

from severstal.evaluate_ensemble import average_probabilities, compare_rows


def test_probability_average_precedes_threshold_and_is_not_logit_average():
    values = {name: torch.tensor([value], dtype=torch.float16) for name, value in zip("ABC", [.99, .2, .2])}
    mean = average_probabilities(values, "ABC")
    assert mean.dtype == torch.float32
    assert mean.item() == pytest.approx((values["A"].item() + values["B"].item() + values["C"].item()) / 3)
    assert not (mean > .5).item()
    assert (torch.stack([torch.logit(v.float()) for v in values.values()]).mean(0).sigmoid() > .5).item()
    assert torch.equal(mean, average_probabilities(values, "CBA"))


def test_probability_average_rejects_misaligned_shapes():
    with pytest.raises(ValueError, match="shapes"):
        average_probabilities({"A": torch.zeros(1, 4, 2, 3), "B": torch.zeros(1, 4, 3, 2)}, "AB")


def test_member_comparison_separates_rescues_losses_and_false_positives():
    base = [{"image": str(i), "class_id": 2, "true_area": target, "pred_area": pred, "dice": dice}
            for i, (target, pred, dice) in enumerate([(10, 0, 0), (10, 5, .5), (0, 2, 0), (0, 0, 1)])]
    current = [{**base[0], "pred_area": 4, "dice": .4}, {**base[1], "pred_area": 0, "dice": 0},
               {**base[2], "pred_area": 0, "dice": 1}, {**base[3], "pred_area": 2, "dice": 0}]
    counts, changes = compare_rows(list(reversed(current)), base)
    assert counts["2"] == {"gained_overlap": 1, "lost_overlap": 1, "added_false_positives": 1, "removed_false_positives": 1}
    assert len(changes) == 4
    with pytest.raises(ValueError, match="identities"):
        compare_rows([current[0]] * 4, base)
