from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import TrainingPatchDataset


class DiceLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(logits).flatten(start_dim=1)
        target = target.flatten(start_dim=1)
        intersection = (probability * target).sum(dim=1)
        dice = (
            2.0 * intersection + self.epsilon
        ) / (
            probability.sum(dim=1) + target.sum(dim=1) + self.epsilon
        )
        return 1.0 - dice.mean()


class BCEOnlyLoss(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, target, thin_mask=None):
        return self.bce(logits, target)


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target, thin_mask=None):
        return (
            self.bce_weight * self.bce(logits, target)
            + self.dice_weight * self.dice(logits, target)
        )


def soft_erode(image: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-image, (3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-image, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(p1, p2)


def soft_dilate(image: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(image, 3, stride=1, padding=1)


def soft_open(image: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(image))


def soft_skeletonize(image: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    image = torch.clamp(image, 0.0, 1.0)
    skeleton = F.relu(image - soft_open(image))
    for _ in range(iterations):
        image = soft_erode(image)
        opened = soft_open(image)
        delta = F.relu(image - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return torch.clamp(skeleton, 0.0, 1.0)


def soft_cldice_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    iterations: int = 10,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    skeleton_pred = soft_skeletonize(probability, iterations)
    skeleton_true = soft_skeletonize(target, iterations)

    topology_precision = (
        (skeleton_pred * target).sum() + epsilon
    ) / (
        skeleton_pred.sum() + epsilon
    )

    topology_sensitivity = (
        (skeleton_true * probability).sum() + epsilon
    ) / (
        skeleton_true.sum() + epsilon
    )

    cldice = (
        2.0 * topology_precision * topology_sensitivity + epsilon
    ) / (
        topology_precision + topology_sensitivity + epsilon
    )

    return 1.0 - cldice


class BCEClDiceLoss(nn.Module):
    def __init__(self, beta: float = 0.5, iterations: int = 10):
        super().__init__()
        self.beta = beta
        self.iterations = iterations
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target, thin_mask=None):
        probability = torch.sigmoid(logits)
        return (
            self.bce(logits, target)
            + self.beta * soft_cldice_loss(
                probability,
                target,
                self.iterations,
            )
        )


class ThinAuxLoss(nn.Module):
    def __init__(self, alpha: float = 0.4, epsilon: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.main_bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target, thin_mask=None):
        if thin_mask is None:
            raise ValueError("Thin mask is required.")

        main_loss = self.main_bce(logits, target)
        pixel_bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )

        thin_loss = (
            pixel_bce * thin_mask
        ).sum() / (
            thin_mask.sum() + self.epsilon
        )

        return main_loss + self.alpha * thin_loss


def estimate_positive_weight(dataset: TrainingPatchDataset, epsilon: float = 1e-8) -> float:
    positive = 0.0
    negative = 0.0

    for loaded in dataset.images:
        positive += float((loaded.mask == 1).sum())
        negative += float((loaded.mask == 0).sum())

    return negative / (positive + epsilon)


def build_loss(
    loss_name: str,
    args,
    training_dataset: TrainingPatchDataset,
    device: torch.device,
) -> Tuple[nn.Module, str]:
    if loss_name == "bce":
        return BCEOnlyLoss(), "BCE"

    if loss_name == "bce_dice":
        return BCEDiceLoss(args.bce_weight, args.dice_weight), "BCE+Dice"

    if loss_name == "weighted_bce":
        value = (
            args.pos_weight
            if args.pos_weight is not None
            else estimate_positive_weight(training_dataset)
        )
        pos_weight = torch.tensor([value], dtype=torch.float32, device=device)
        return BCEOnlyLoss(pos_weight), f"Weighted BCE ({value:.4f})"

    if loss_name == "bce_cldice":
        return BCEClDiceLoss(
            args.cldice_beta,
            args.cldice_iterations,
        ), "BCE+clDice"

    if loss_name == "thinaux_fixed":
        return ThinAuxLoss(args.alpha), "Fixed ThinAux"

    if loss_name == "thinaux_adaptive":
        return ThinAuxLoss(args.alpha), "Adaptive ThinAux"

    raise ValueError(f"Unknown loss: {loss_name}")
