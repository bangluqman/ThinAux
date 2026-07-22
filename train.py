import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import config as cfg
from dataset import PatchDataset, pair_files, positions, read_image, read_mask, split_train_val
from loss_metric import BCEClDice, BCEDice, BCEOnly, ThinAuxLoss
from loss_metric import auc_value, cldice_score, dice_score, precision_score
from loss_metric import sensitivity_score, specificity_score, thin_recall
from model import UNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if cfg.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_thin_type(loss_name):
    if loss_name == "thinaux_fixed":
        return "fixed"
    if loss_name == "thinaux_adaptive":
        return "adaptive"
    return "none"


def count_pos_weight(dataset):
    pos = 0
    neg = 0
    for _, _, mask, _ in dataset.images:
        pos += (mask == 1).sum()
        neg += (mask == 0).sum()
    return float(neg / (pos + 1e-8))


def get_loss(loss_name, dataset, device):
    if loss_name == "bce":
        return BCEOnly(), "BCE"
    if loss_name == "bce_dice":
        return BCEDice(), "BCE+Dice"
    if loss_name == "bce_cldice":
        return BCEClDice(), "BCE+clDice"
    if loss_name == "weighted_bce":
        value = count_pos_weight(dataset)
        weight = torch.tensor([value], dtype=torch.float32, device=device)
        return BCEOnly(weight), "Weighted BCE"
    if loss_name == "thinaux_fixed":
        return ThinAuxLoss(cfg.ALPHA), "Fixed ThinAux"
    if loss_name == "thinaux_adaptive":
        return ThinAuxLoss(cfg.ALPHA), "Adaptive ThinAux"
    raise ValueError(loss_name)


def predict_image(model, img, device):
    patch_size = cfg.PATCH_SIZE
    stride = cfg.STRIDE
    h0, w0 = img.shape[:2]
    ph = max(0, patch_size - h0)
    pw = max(0, patch_size - w0)

    if ph > 0 or pw > 0:
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")

    h, w = img.shape[:2]
    result = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    patches = []
    coords = []

    def run_batch():
        if len(patches) == 0:
            return
        batch = torch.stack(patches).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(batch)).cpu().numpy()[:, 0]
        for p, (y, x) in zip(prob, coords):
            result[y:y + patch_size, x:x + patch_size] += p
            count[y:y + patch_size, x:x + patch_size] += 1
        patches.clear()
        coords.clear()

    for y in positions(h, patch_size, stride):
        for x in positions(w, patch_size, stride):
            patch = img[y:y + patch_size, x:x + patch_size]
            patch = torch.tensor(patch.transpose(2, 0, 1), dtype=torch.float32)
            patches.append(patch)
            coords.append((y, x))
            if len(patches) == 4:
                run_batch()

    run_batch()
    result = result / np.maximum(count, 1)
    return result[:h0, :w0]


def evaluate(model, data, device):
    rows = []
    model.eval()

    for name, img_path, mask_path in data:
        img = read_image(img_path)
        gt = read_mask(mask_path)
        prob = predict_image(model, img, device)
        pred = (prob >= cfg.PRED_THRESHOLD).astype(np.float32)

        rows.append({
            "image": name,
            "Dice": dice_score(pred, gt),
            "Precision": precision_score(pred, gt),
            "Sensitivity": sensitivity_score(pred, gt),
            "Specificity": specificity_score(pred, gt),
            "AUC": auc_value(prob, gt),
            "Thin_Recall": thin_recall(pred, gt, cfg.THIN_THRESHOLD),
            "clDice": cldice_score(pred, gt),
        })

    df = pd.DataFrame(rows)
    result = {}
    for col in ["Dice", "Precision", "Sensitivity", "Specificity", "AUC", "Thin_Recall", "clDice"]:
        result[col] = float(np.nanmean(df[col].values))
    return result, df


def train_one(loss_name, seed, train_data, val_data, test_data):
    set_seed(seed)
    device = get_device()

    dataset = PatchDataset(
        train_data,
        patch_size=cfg.PATCH_SIZE,
        stride=cfg.STRIDE,
        thin_type=get_thin_type(loss_name),
        threshold=cfg.THIN_THRESHOLD,
        percentile=cfg.ADAPTIVE_PERCENTILE,
        augment=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
    )

    model = UNet().to(device)
    criterion, method_name = get_loss(loss_name, dataset, device)
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)

    run_dir = os.path.join(cfg.OUTPUT_DIR, loss_name + "_seed" + str(seed))
    os.makedirs(run_dir, exist_ok=True)

    best_dice = -1
    best_epoch = 0
    history = []

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        losses = []

        for img, mask, thin in loader:
            img = img.to(device)
            mask = mask.to(device)
            thin = thin.to(device)
            optimizer.zero_grad()
            logits = model(img)
            loss = criterion(logits, mask, thin)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses))
        val_result, val_df = evaluate(model, val_data, device)
        history.append({
            "epoch": epoch,
            "loss": train_loss,
            "val_dice": val_result["Dice"],
            "val_thin_recall": val_result["Thin_Recall"],
        })

        print(loss_name, seed, epoch, train_loss, val_result["Dice"])

        if val_result["Dice"] > best_dice:
            best_dice = val_result["Dice"]
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
            val_df.to_csv(os.path.join(run_dir, "best_val_result.csv"), index=False)

    pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)

    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pt"), map_location=device))
    test_result, test_df = evaluate(model, test_data, device)
    test_df.to_csv(os.path.join(run_dir, "test_result_per_image.csv"), index=False)

    test_result["method"] = method_name
    test_result["loss_name"] = loss_name
    test_result["seed"] = seed
    test_result["best_epoch"] = best_epoch
    test_result["best_val_dice"] = best_dice

    with open(os.path.join(run_dir, "test_summary.json"), "w") as f:
        json.dump(test_result, f, indent=2)

    return test_result


def main():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    all_train = pair_files(cfg.TRAIN_IMG, cfg.TRAIN_MASK)
    test_data = pair_files(cfg.TEST_IMG, cfg.TEST_MASK)
    train_data, val_data = split_train_val(all_train, cfg.VAL_RATIO, cfg.SPLIT_SEED)

    with open(os.path.join(cfg.OUTPUT_DIR, "split.json"), "w") as f:
        json.dump({
            "train": [x[0] for x in train_data],
            "val": [x[0] for x in val_data],
            "test": [x[0] for x in test_data],
        }, f, indent=2)

    methods = [
        "bce",
        "bce_dice",
        "bce_cldice",
        "weighted_bce",
        "thinaux_fixed",
        "thinaux_adaptive",
    ]

    results = []

    for method in methods:
        for seed in cfg.SEEDS:
            result = train_one(method, seed, train_data, val_data, test_data)
            results.append(result)
            pd.DataFrame(results).to_csv(os.path.join(cfg.OUTPUT_DIR, "all_results.csv"), index=False)

    df = pd.DataFrame(results)
    df.groupby("method").mean(numeric_only=True).to_csv(os.path.join(cfg.OUTPUT_DIR, "mean_result.csv"))
    df.groupby("method").std(numeric_only=True).to_csv(os.path.join(cfg.OUTPUT_DIR, "std_result.csv"))


if __name__ == "__main__":
    main()
