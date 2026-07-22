import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


class DiceLoss(nn.Module):
    def forward(self, logits, target):
        prob = torch.sigmoid(logits).view(logits.size(0), -1)
        target = target.view(target.size(0), -1)
        inter = (prob * target).sum(1)
        dice = (2 * inter + 1e-8) / (prob.sum(1) + target.sum(1) + 1e-8)
        return 1 - dice.mean()


class BCEDice(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, target, thin=None):
        return 0.5 * self.bce(logits, target) + 0.5 * self.dice(logits, target)


class ThinAuxLoss(nn.Module):
    def __init__(self, alpha=0.4):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target, thin):
        main_loss = self.bce(logits, target)
        pixel_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        thin_loss = (pixel_loss * thin).sum() / (thin.sum() + 1e-8)
        return main_loss + self.alpha * thin_loss


class BCEOnly(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, target, thin=None):
        return self.bce(logits, target)


def soft_erode(x):
    p1 = -F.max_pool2d(-x, (3, 1), 1, (1, 0))
    p2 = -F.max_pool2d(-x, (1, 3), 1, (0, 1))
    return torch.minimum(p1, p2)


def soft_dilate(x):
    return F.max_pool2d(x, 3, 1, 1)


def soft_open(x):
    return soft_dilate(soft_erode(x))


def soft_skeleton(x, n=10):
    x = torch.clamp(x, 0, 1)
    skel = F.relu(x - soft_open(x))
    for _ in range(n):
        x = soft_erode(x)
        opened = soft_open(x)
        delta = F.relu(x - opened)
        skel = skel + F.relu(delta - skel * delta)
    return torch.clamp(skel, 0, 1)


def cldice_loss(prob, target, n=10):
    sp = soft_skeleton(prob, n)
    st = soft_skeleton(target, n)
    tprec = ((sp * target).sum() + 1e-8) / (sp.sum() + 1e-8)
    tsens = ((st * prob).sum() + 1e-8) / (st.sum() + 1e-8)
    score = (2 * tprec * tsens + 1e-8) / (tprec + tsens + 1e-8)
    return 1 - score


class BCEClDice(nn.Module):
    def __init__(self, beta=0.5):
        super().__init__()
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target, thin=None):
        prob = torch.sigmoid(logits)
        return self.bce(logits, target) + self.beta * cldice_loss(prob, target)


def dice_score(pred, gt):
    inter = (pred * gt).sum()
    return float((2 * inter + 1e-8) / (pred.sum() + gt.sum() + 1e-8))


def precision_score(pred, gt):
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    return 0.0 if tp + fp == 0 else float(tp / (tp + fp))


def sensitivity_score(pred, gt):
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    return np.nan if tp + fn == 0 else float(tp / (tp + fn))


def specificity_score(pred, gt):
    tn = np.logical_and(pred == 0, gt == 0).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    return np.nan if tn + fp == 0 else float(tn / (tn + fp))


def thin_recall(pred, gt, threshold=1.5):
    m = gt.astype(np.uint8)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    thin = ((dist <= threshold) & (m == 1)).astype(np.float32)
    return np.nan if thin.sum() == 0 else float((pred * thin).sum() / thin.sum())


def auc_value(prob, gt):
    return np.nan if len(np.unique(gt)) < 2 else float(roc_auc_score(gt.reshape(-1), prob.reshape(-1)))


def cldice_score(pred, gt):
    p = torch.tensor(pred[None, None], dtype=torch.float32)
    g = torch.tensor(gt[None, None], dtype=torch.float32)
    with torch.no_grad():
        sp = soft_skeleton(p)
        st = soft_skeleton(g)
        tprec = ((sp * g).sum() + 1e-8) / (sp.sum() + 1e-8)
        tsens = ((st * p).sum() + 1e-8) / (st.sum() + 1e-8)
        score = (2 * tprec * tsens + 1e-8) / (tprec + tsens + 1e-8)
    return float(score)
