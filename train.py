import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data import TrainingPatchDataset
from inference import evaluate_records
from losses import build_loss
from model import StandardUNet
from utils import save_json, seed_worker, set_seed


def thin_mode(loss_name: str) -> str:
    if loss_name == "thinaux_fixed":
        return "fixed"
    if loss_name == "thinaux_adaptive":
        return "adaptive"
    return "none"


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []

    for images, targets, thin_masks in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        thin_masks = thin_masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets, thin_masks)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss: {loss.item()}")

        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return float(np.mean(losses))


def train_single_run(
    args,
    loss_name,
    seed,
    train_records,
    val_records,
    test_records,
):
    set_seed(seed)

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
        if args.device != "auto"
        else "cpu"
    )

    training_dataset = TrainingPatchDataset(
        train_records,
        args.patch_size,
        args.train_stride,
        thin_mode(loss_name),
        args.fixed_threshold,
        args.adaptive_percentile,
        args.augment,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )

    model = StandardUNet().to(device)
    criterion, display_name = build_loss(
        loss_name,
        args,
        training_dataset,
        device,
    )
    criterion = criterion.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    run_name = f"{args.dataset}_{loss_name}_seed{seed}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config.update(
        {
            "loss_name": loss_name,
            "display_name": display_name,
            "seed": seed,
            "resolved_device": str(device),
            "train_images": [record.sample_id for record in train_records],
            "val_images": [record.sample_id for record in val_records],
            "test_images": [record.sample_id for record in test_records],
            "training_patch_count": len(training_dataset),
        }
    )
    save_json(config, run_dir / "config.json")

    history_rows = []
    best_val_dice = -math.inf
    best_epoch = -1
    best_checkpoint_path = run_dir / "best_model.pt"
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        if epoch % args.validation_every == 0 or epoch == args.epochs:
            val_metrics, val_per_image = evaluate_records(
                model,
                val_records,
                device,
                args.patch_size,
                args.inference_stride,
                args.inference_batch_size,
                args.prediction_threshold,
                args.evaluation_thin_threshold,
                args.cldice_iterations,
            )

            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **{f"val_{key}": value for key, value in val_metrics.items()},
                    "elapsed_seconds": time.time() - start_time,
                }
            )

            print(
                f"{run_name} | {epoch:03d}/{args.epochs} | "
                f"{train_loss:.6f} | {val_metrics['Dice']:.4f} | "
                f"{val_metrics['Thin_Recall']:.4f}"
            )

            if val_metrics["Dice"] > best_val_dice:
                best_val_dice = val_metrics["Dice"]
                best_epoch = epoch

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "validation_metrics": val_metrics,
                        "config": config,
                    },
                    best_checkpoint_path,
                )

                val_per_image.to_csv(
                    run_dir / "best_validation_per_image.csv",
                    index=False,
                )

    pd.DataFrame(history_rows).to_csv(
        run_dir / "training_history.csv",
        index=False,
    )

    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_metrics, test_per_image = evaluate_records(
        model,
        test_records,
        device,
        args.patch_size,
        args.inference_stride,
        args.inference_batch_size,
        args.prediction_threshold,
        args.evaluation_thin_threshold,
        args.cldice_iterations,
        run_dir / "test_predictions" if args.save_predictions else None,
    )

    test_per_image.to_csv(
        run_dir / "test_metrics_per_image.csv",
        index=False,
    )

    result = {
        "dataset": args.dataset,
        "loss_key": loss_name,
        "method": display_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_Dice": best_val_dice,
        **test_metrics,
    }

    save_json(result, run_dir / "test_summary.json")
    return result
