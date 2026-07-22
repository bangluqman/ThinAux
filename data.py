import glob
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ImageRecord:
    sample_id: str
    image_path: str
    mask_path: str


@dataclass
class LoadedImage:
    sample_id: str
    image: np.ndarray
    mask: np.ndarray
    thin_mask: np.ndarray


def natural_key(path: str) -> List[object]:
    name = Path(path).name.lower()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def infer_sample_id(path: str) -> str:
    stem = Path(path).stem.lower()
    suffix_patterns = [
        r"[._-](training|test)$",
        r"[._-](manual1|manual2|1st_manual|2nd_manual)$",
        r"[._-](1stho|2ndho)$",
        r"[._-](mask|masks|label|labels|annotation|annotations)$",
        r"[._-](ah|vk)$",
    ]
    changed = True
    while changed:
        changed = False
        for pattern in suffix_patterns:
            updated = re.sub(pattern, "", stem, flags=re.IGNORECASE)
            if updated != stem:
                stem = updated
                changed = True
    sample_id = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    if not sample_id:
        raise ValueError(f"Could not infer a sample ID from: {path}")
    return sample_id


def discover_records(image_glob: str, mask_glob: str) -> List[ImageRecord]:
    image_paths = sorted(glob.glob(image_glob), key=natural_key)
    mask_paths = sorted(glob.glob(mask_glob), key=natural_key)

    if not image_paths:
        raise FileNotFoundError(f"No images found for: {image_glob}")
    if not mask_paths:
        raise FileNotFoundError(f"No masks found for: {mask_glob}")

    image_map: Dict[str, str] = {}
    mask_map: Dict[str, str] = {}

    for path in image_paths:
        sample_id = infer_sample_id(path)
        if sample_id in image_map:
            raise ValueError(f"Duplicate inferred image ID: {sample_id}")
        image_map[sample_id] = path

    for path in mask_paths:
        sample_id = infer_sample_id(path)
        if sample_id in mask_map:
            raise ValueError(f"Duplicate inferred mask ID: {sample_id}")
        mask_map[sample_id] = path

    common_ids = sorted(set(image_map) & set(mask_map))
    if len(common_ids) != len(image_paths) or len(common_ids) != len(mask_paths):
        missing_masks = sorted(set(image_map) - set(mask_map))
        missing_images = sorted(set(mask_map) - set(image_map))
        raise ValueError(
            f"Images without masks: {missing_masks}; masks without images: {missing_images}"
        )

    return [
        ImageRecord(sample_id=sid, image_path=image_map[sid], mask_path=mask_map[sid])
        for sid in common_ids
    ]


def load_split_json(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as handle:
        split = json.load(handle)
    required = {"train", "val", "test"}
    missing = required - set(split)
    if missing:
        raise ValueError(f"Split JSON is missing keys: {sorted(missing)}")
    return {key: [str(item) for item in split[key]] for key in required}


def select_records(records: Sequence[ImageRecord], ids: Sequence[str]) -> List[ImageRecord]:
    record_map = {record.sample_id: record for record in records}
    missing = [sample_id for sample_id in ids if sample_id not in record_map]
    if missing:
        raise ValueError(f"Unknown sample IDs: {missing}")
    return [record_map[sample_id] for sample_id in ids]


def deterministic_split(
    records: Sequence[ImageRecord],
    val_fraction: float,
    test_fraction: float,
    split_seed: int,
) -> Tuple[List[ImageRecord], List[ImageRecord], List[ImageRecord]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0, 1).")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be less than 1.")

    records = list(records)
    rng = np.random.default_rng(split_seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)

    n_total = len(records)
    n_test = int(round(n_total * test_fraction))
    n_val = int(round(n_total * val_fraction))

    if test_fraction > 0 and n_test == 0:
        n_test = 1
    if n_val == 0:
        n_val = 1
    if n_total - n_val - n_test < 1:
        raise ValueError("Dataset is too small for the requested split fractions.")

    test_idx = set(indices[:n_test].tolist())
    val_idx = set(indices[n_test:n_test + n_val].tolist())
    train_idx = set(indices[n_test + n_val:].tolist())

    train = [records[i] for i in range(n_total) if i in train_idx]
    val = [records[i] for i in range(n_total) if i in val_idx]
    test = [records[i] for i in range(n_total) if i in test_idx]
    return train, val, test


def resolve_dataset_splits(args):
    if args.dataset == "drive":
        official_train = discover_records(args.train_img, args.train_mask)
        official_test = discover_records(args.test_img, args.test_mask)
        if args.split_json:
            split = load_split_json(args.split_json)
            train_records = select_records(official_train, split["train"])
            val_records = select_records(official_train, split["val"])
            test_records = official_test
        else:
            train_records, val_records, _ = deterministic_split(
                official_train,
                args.val_fraction,
                0.0,
                args.split_seed,
            )
            test_records = official_test
    else:
        all_records = discover_records(args.all_img, args.all_mask)
        if args.split_json:
            split = load_split_json(args.split_json)
            train_records = select_records(all_records, split["train"])
            val_records = select_records(all_records, split["val"])
            test_records = select_records(all_records, split["test"])
        else:
            train_records, val_records, test_records = deterministic_split(
                all_records,
                args.val_fraction,
                args.test_fraction,
                args.split_seed,
            )

    if not train_records or not val_records or not test_records:
        raise ValueError("Empty dataset split.")
    return train_records, val_records, test_records


def save_split_manifest(path: Path, train_records, val_records, test_records) -> None:
    manifest = {
        "train": [record.sample_id for record in train_records],
        "val": [record.sample_id for record in val_records],
        "test": [record.sample_id for record in test_records],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def read_rgb_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def read_binary_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    return (mask > 127).astype(np.float32)


def distance_transform(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0.5).astype(np.uint8)
    if mask_u8.sum() == 0:
        return np.zeros_like(mask, dtype=np.float32)
    return cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3).astype(np.float32)


def fixed_thin_mask(mask: np.ndarray, threshold: float = 1.5) -> np.ndarray:
    mask_bin = (mask > 0.5).astype(np.float32)
    dist = distance_transform(mask_bin)
    return ((dist <= threshold) & (mask_bin == 1)).astype(np.float32)


def adaptive_thin_mask(mask: np.ndarray, percentile: float = 30.0) -> np.ndarray:
    mask_bin = (mask > 0.5).astype(np.float32)
    dist = distance_transform(mask_bin)
    vessel_distances = dist[mask_bin == 1]
    if vessel_distances.size == 0:
        return np.zeros_like(mask_bin, dtype=np.float32)
    tau = float(np.percentile(vessel_distances, percentile))
    return ((dist <= tau) & (mask_bin == 1)).astype(np.float32)


def make_thin_mask(mask, mode, fixed_threshold, adaptive_percentile):
    if mode == "fixed":
        return fixed_thin_mask(mask, fixed_threshold)
    if mode == "adaptive":
        return adaptive_thin_mask(mask, adaptive_percentile)
    if mode == "none":
        return np.zeros_like(mask, dtype=np.float32)
    raise ValueError(f"Unknown thin-mask mode: {mode}")


def sliding_positions(length: int, patch_size: int, stride: int) -> List[int]:
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def pad_to_patch_size(image, mask, thin_mask, patch_size):
    original_h, original_w = mask.shape
    pad_h = max(0, patch_size - original_h)
    pad_w = max(0, patch_size - original_w)

    if pad_h == 0 and pad_w == 0:
        return image, mask, thin_mask, (original_h, original_w)

    image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")

    if thin_mask is not None:
        thin_mask = np.pad(thin_mask, ((0, pad_h), (0, pad_w)), mode="constant")

    return image, mask, thin_mask, (original_h, original_w)


class TrainingPatchDataset(Dataset):
    def __init__(
        self,
        records,
        patch_size,
        stride,
        thin_mode,
        fixed_threshold,
        adaptive_percentile,
        augment,
    ):
        self.patch_size = patch_size
        self.augment = augment
        self.images: List[LoadedImage] = []
        self.patch_index: List[Tuple[int, int, int]] = []

        for record in records:
            image = read_rgb_image(record.image_path)
            mask = read_binary_mask(record.mask_path)

            if image.shape[:2] != mask.shape:
                raise ValueError(f"Size mismatch for {record.sample_id}")

            thin = make_thin_mask(
                mask,
                thin_mode,
                fixed_threshold,
                adaptive_percentile,
            )

            image, mask, thin, _ = pad_to_patch_size(
                image,
                mask,
                thin,
                patch_size,
            )

            image_idx = len(self.images)
            self.images.append(LoadedImage(record.sample_id, image, mask, thin))

            h, w = mask.shape
            for y in sliding_positions(h, patch_size, stride):
                for x in sliding_positions(w, patch_size, stride):
                    self.patch_index.append((image_idx, y, x))

    def __len__(self) -> int:
        return len(self.patch_index)

    def __getitem__(self, index: int):
        image_idx, y, x = self.patch_index[index]
        loaded = self.images[image_idx]
        ps = self.patch_size

        image = loaded.image[y:y + ps, x:x + ps].copy()
        mask = loaded.mask[y:y + ps, x:x + ps].copy()
        thin = loaded.thin_mask[y:y + ps, x:x + ps].copy()

        if self.augment:
            if random.random() < 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
                thin = np.flip(thin, axis=1).copy()

            if random.random() < 0.5:
                image = np.flip(image, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()
                thin = np.flip(thin, axis=0).copy()

            k = random.choice([0, 1, 2, 3])
            if k:
                image = np.rot90(image, k, axes=(0, 1)).copy()
                mask = np.rot90(mask, k, axes=(0, 1)).copy()
                thin = np.rot90(thin, k, axes=(0, 1)).copy()

        image_t = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        mask_t = torch.from_numpy(mask[None]).float()
        thin_t = torch.from_numpy(thin[None]).float()
        return image_t, mask_t, thin_t
