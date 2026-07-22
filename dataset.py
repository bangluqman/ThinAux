import glob
import os
import random
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def file_id(path):
    name = os.path.splitext(os.path.basename(path))[0].lower()
    name = re.sub(r"[_\.-](training|test|manual1|manual2|1st_manual|2nd_manual|mask|label)$", "", name)
    return name


def pair_files(img_pattern, mask_pattern):
    imgs = sorted(glob.glob(img_pattern))
    masks = sorted(glob.glob(mask_pattern))
    img_dict = {file_id(x): x for x in imgs}
    mask_dict = {file_id(x): x for x in masks}
    ids = sorted(set(img_dict) & set(mask_dict))
    if len(ids) != len(imgs) or len(ids) != len(masks):
        raise RuntimeError("image and mask pairing error")
    return [(i, img_dict[i], mask_dict[i]) for i in ids]


def split_train_val(data, val_ratio=0.2, seed=2026):
    ids = list(range(len(data)))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(data) * val_ratio)))
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]
    return [data[i] for i in train_ids], [data[i] for i in val_ids]


def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def read_mask(path):
    mask = cv2.imread(path, 0)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask > 127).astype(np.float32)


def thin_mask_fixed(mask, threshold=1.5):
    m = (mask > 0.5).astype(np.uint8)
    if m.sum() == 0:
        return np.zeros_like(mask, dtype=np.float32)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    return ((dist <= threshold) & (m == 1)).astype(np.float32)


def thin_mask_adaptive(mask, percentile=30):
    m = (mask > 0.5).astype(np.uint8)
    if m.sum() == 0:
        return np.zeros_like(mask, dtype=np.float32)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    values = dist[m == 1]
    if len(values) == 0:
        return np.zeros_like(mask, dtype=np.float32)
    threshold = np.percentile(values, percentile)
    return ((dist <= threshold) & (m == 1)).astype(np.float32)


def positions(size, patch_size, stride):
    if size <= patch_size:
        return [0]
    p = list(range(0, size - patch_size + 1, stride))
    if p[-1] != size - patch_size:
        p.append(size - patch_size)
    return p


def pad_data(img, mask, thin, patch_size):
    h, w = mask.shape
    ph = max(0, patch_size - h)
    pw = max(0, patch_size - w)
    if ph == 0 and pw == 0:
        return img, mask, thin
    img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    mask = np.pad(mask, ((0, ph), (0, pw)), mode="constant")
    thin = np.pad(thin, ((0, ph), (0, pw)), mode="constant")
    return img, mask, thin


class PatchDataset(Dataset):
    def __init__(self, data, patch_size=256, stride=128, thin_type="none", threshold=1.5, percentile=30, augment=False):
        self.patch_size = patch_size
        self.augment = augment
        self.images = []
        self.samples = []

        for name, img_path, mask_path in data:
            img = read_image(img_path)
            mask = read_mask(mask_path)
            if img.shape[:2] != mask.shape:
                raise RuntimeError(name)

            if thin_type == "fixed":
                thin = thin_mask_fixed(mask, threshold)
            elif thin_type == "adaptive":
                thin = thin_mask_adaptive(mask, percentile)
            else:
                thin = np.zeros_like(mask, dtype=np.float32)

            img, mask, thin = pad_data(img, mask, thin, patch_size)
            idx = len(self.images)
            self.images.append((name, img, mask, thin))
            h, w = mask.shape

            for y in positions(h, patch_size, stride):
                for x in positions(w, patch_size, stride):
                    self.samples.append((idx, y, x))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        idx, y, x = self.samples[index]
        _, img, mask, thin = self.images[idx]
        ps = self.patch_size

        img = img[y:y + ps, x:x + ps].copy()
        mask = mask[y:y + ps, x:x + ps].copy()
        thin = thin[y:y + ps, x:x + ps].copy()

        if self.augment:
            if random.random() < 0.5:
                img = np.flip(img, 1).copy()
                mask = np.flip(mask, 1).copy()
                thin = np.flip(thin, 1).copy()
            if random.random() < 0.5:
                img = np.flip(img, 0).copy()
                mask = np.flip(mask, 0).copy()
                thin = np.flip(thin, 0).copy()
            k = random.randint(0, 3)
            if k > 0:
                img = np.rot90(img, k).copy()
                mask = np.rot90(mask, k).copy()
                thin = np.rot90(thin, k).copy()

        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)
        mask = torch.tensor(mask[None], dtype=torch.float32)
        thin = torch.tensor(thin[None], dtype=torch.float32)
        return img, mask, thin
