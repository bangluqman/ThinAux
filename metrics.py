import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from data import fixed_thin_mask
from losses import soft_skeletonize


def dice_score(prediction, target, epsilon=1e-8):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    denominator = prediction.sum() + target.sum()
    if denominator == 0:
        return 1.0
    intersection = np.logical_and(prediction, target).sum()
    return float((2.0 * intersection + epsilon) / (denominator + epsilon))


def precision_score(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(prediction, target).sum()
    fp = np.logical_and(prediction, np.logical_not(target)).sum()
    denominator = tp + fp
    return 0.0 if denominator == 0 else float(tp / denominator)


def sensitivity_score(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(prediction, target).sum()
    fn = np.logical_and(np.logical_not(prediction), target).sum()
    denominator = tp + fn
    return float("nan") if denominator == 0 else float(tp / denominator)


def specificity_score(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    tn = np.logical_and(np.logical_not(prediction), np.logical_not(target)).sum()
    fp = np.logical_and(prediction, np.logical_not(target)).sum()
    denominator = tn + fp
    return float("nan") if denominator == 0 else float(tn / denominator)


def thin_recall_score(prediction, target, threshold):
    thin = fixed_thin_mask(target, threshold).astype(bool)
    denominator = thin.sum()
    if denominator == 0:
        return float("nan")
    recovered = np.logical_and(prediction.astype(bool), thin).sum()
    return float(recovered / denominator)


def cldice_score(prediction, target, iterations=10, epsilon=1e-8):
    prediction_t = torch.from_numpy(prediction[None, None].astype(np.float32))
    target_t = torch.from_numpy(target[None, None].astype(np.float32))

    with torch.no_grad():
        skeleton_pred = soft_skeletonize(prediction_t, iterations)
        skeleton_true = soft_skeletonize(target_t, iterations)

        topology_precision = (
            (skeleton_pred * target_t).sum() + epsilon
        ) / (
            skeleton_pred.sum() + epsilon
        )

        topology_sensitivity = (
            (skeleton_true * prediction_t).sum() + epsilon
        ) / (
            skeleton_true.sum() + epsilon
        )

        value = (
            2.0 * topology_precision * topology_sensitivity + epsilon
        ) / (
            topology_precision + topology_sensitivity + epsilon
        )

    return float(value.item())


def auc_score(target, probability):
    if np.unique(target).size < 2:
        return float("nan")
    return float(roc_auc_score(target.reshape(-1), probability.reshape(-1)))
