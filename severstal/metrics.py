"""Competition mean Dice is averaged over image/class pairs, not pooled pixels."""

import numpy as np
import torch


class DiceMeter:
    def __init__(self):
        self.rows = []

    def update(self, predictions, targets, names):
        predictions, targets = predictions.bool(), targets.bool()
        areas_p = predictions.flatten(2).sum(2)
        areas_t = targets.flatten(2).sum(2)
        intersection = (predictions & targets).flatten(2).sum(2)
        denominator = areas_p + areas_t
        dice = torch.where(denominator > 0, 2.0 * intersection / denominator.clamp_min(1), 1.0)
        for name, scores, pred_area, true_area in zip(names, dice.cpu().tolist(), areas_p.cpu().tolist(), areas_t.cpu().tolist()):
            for c in range(4):
                self.rows.append({"image": name, "class_id": c + 1, "dice": scores[c],
                                  "pred_area": pred_area[c], "true_area": true_area[c]})

    def summary(self):
        result = {"dice": float(np.mean([r["dice"] for r in self.rows]))}
        nonempty = [r["dice"] for r in self.rows if r["true_area"] > 0]
        result["nonempty_dice"] = float(np.mean(nonempty)) if nonempty else None
        result["classes"] = {}
        for c in range(1, 5):
            rows = [r for r in self.rows if r["class_id"] == c]
            positives = [r for r in rows if r["true_area"] > 0]
            tp = sum(r["pred_area"] > 0 for r in positives)
            fp = sum(r["pred_area"] > 0 and r["true_area"] == 0 for r in rows)
            result["classes"][str(c)] = {
                "dice": float(np.mean([r["dice"] for r in rows])),
                "nonempty_dice": float(np.mean([r["dice"] for r in positives])) if positives else None,
                "image_class_recall": tp / len(positives) if positives else None,
                "false_positive_pairs": fp, "false_negative_pairs": len(positives) - tp,
            }
        return result


def segmentation_loss(logits, targets, positive_weights, bce_weight=0.75):
    weights = logits.new_tensor(positive_weights).reshape(4, 1, 1)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=weights)
    probability = logits.sigmoid().float().flatten(2)
    truth = targets.float().flatten(2)
    dice = 1 - (2 * (probability * truth).sum(2) + 1) / (probability.sum(2) + truth.sum(2) + 1)
    positive = truth.sum(2) > 0
    dice = (dice * positive).sum() / positive.sum().clamp_min(1)
    return bce_weight * bce + (1 - bce_weight) * dice
