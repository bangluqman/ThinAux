from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

from data import (
    ImageRecord,
    pad_to_patch_size,
    read_binary_mask,
    read_rgb_image,
    sliding_positions,
)
from metrics import (
    auc_score,
    cldice_score,
    dice_score,
    precision_score,
    sensitivity_score,
    specificity_score,
    thin_recall_score,
)


def predict_full_image(
    model,
    image,
    device,
    patch_size,
    stride,
    inference_batch_size,
):
    dummy_mask = np.zeros(image.shape[:2], dtype=np.float32)
    padded_image, _, _, original_shape = pad_to_patch_size(
        image,
        dummy_mask,
        None,
        patch_size,
    )

    h, w = padded_image.shape[:2]
    ys = sliding_positions(h, patch_size, stride)
    xs = sliding_positions(w, patch_size, stride)

    probability_sum = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    patch_tensors: List[torch.Tensor] = []
    coordinates: List[Tuple[int, int]] = []

    def flush():
        if not patch_tensors:
            return

        batch = torch.stack(patch_tensors).to(device, non_blocking=True)

        with torch.no_grad():
            probability = torch.sigmoid(model(batch)).cpu().numpy()[:, 0]

        for patch, (y, x) in zip(probability, coordinates):
            probability_sum[y:y + patch_size, x:x + patch_size] += patch
            count_map[y:y + patch_size, x:x + patch_size] += 1.0

        patch_tensors.clear()
        coordinates.clear()

    for y in ys:
        for x in xs:
            patch = padded_image[y:y + patch_size, x:x + patch_size]
            patch_t = torch.from_numpy(np.transpose(patch, (2, 0, 1))).float()
            patch_tensors.append(patch_t)
            coordinates.append((y, x))

            if len(patch_tensors) >= inference_batch_size:
                flush()

    flush()

    probability_map = probability_sum / np.maximum(count_map, 1.0)
    original_h, original_w = original_shape
    return probability_map[:original_h, :original_w]


def evaluate_records(
    model,
    records,
    device,
    patch_size,
    inference_stride,
    inference_batch_size,
    prediction_threshold,
    evaluation_thin_threshold,
    cldice_iterations,
    save_predictions_dir: Optional[Path] = None,
):
    model.eval()
    rows = []

    if save_predictions_dir is not None:
        save_predictions_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        image = read_rgb_image(record.image_path)
        target = read_binary_mask(record.mask_path)

        probability = predict_full_image(
            model,
            image,
            device,
            patch_size,
            inference_stride,
            inference_batch_size,
        )

        prediction = (probability >= prediction_threshold).astype(np.float32)

        row = {
            "sample_id": record.sample_id,
            "Dice": dice_score(prediction, target),
            "Precision": precision_score(prediction, target),
            "Sensitivity": sensitivity_score(prediction, target),
            "Specificity": specificity_score(prediction, target),
            "AUC": auc_score(target, probability),
            "Thin_Recall": thin_recall_score(
                prediction,
                target,
                evaluation_thin_threshold,
            ),
            "clDice": cldice_score(
                prediction,
                target,
                cldice_iterations,
            ),
        }

        rows.append(row)

        if save_predictions_dir is not None:
            cv2.imwrite(
                str(save_predictions_dir / f"{record.sample_id}_probability.png"),
                np.clip(probability * 255.0, 0, 255).astype(np.uint8),
            )
            cv2.imwrite(
                str(save_predictions_dir / f"{record.sample_id}_prediction.png"),
                (prediction * 255).astype(np.uint8),
            )

    per_image = pd.DataFrame(rows)
    metric_columns = [
        "Dice",
        "Precision",
        "Sensitivity",
        "Specificity",
        "AUC",
        "Thin_Recall",
        "clDice",
    ]

    macro = {
        metric: float(np.nanmean(per_image[metric].to_numpy(dtype=np.float64)))
        for metric in metric_columns
    }

    return macro, per_image
