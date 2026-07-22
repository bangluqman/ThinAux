from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import resolve_dataset_splits, save_split_manifest
from train import train_single_run


def loss_list(args):
    if args.experiment == "single":
        return [args.loss]
    return [
        "bce",
        "bce_dice",
        "bce_cldice",
        "weighted_bce",
        "thinaux_fixed",
        "thinaux_adaptive",
    ]


def make_summary(results: pd.DataFrame, output_dir: Path) -> None:
    metric_columns = [
        "Dice",
        "Precision",
        "Sensitivity",
        "Specificity",
        "AUC",
        "Thin_Recall",
        "clDice",
        "best_epoch",
    ]

    mean_table = results.groupby("method")[metric_columns].mean().reset_index()
    std_table = results.groupby("method")[metric_columns].std().reset_index()

    mean_table.to_csv(output_dir / "summary_mean.csv", index=False)
    std_table.to_csv(output_dir / "summary_std.csv", index=False)

    formatted_rows = []

    for method in mean_table["method"]:
        mean_row = mean_table[mean_table["method"] == method].iloc[0]
        std_row = std_table[std_table["method"] == method].iloc[0]
        row = {"Method": method}

        for metric in metric_columns[:-1]:
            row[metric] = f"{mean_row[metric]:.4f} ± {std_row[metric]:.4f}"

        row["Best_Epoch"] = (
            f"{mean_row['best_epoch']:.1f} ± {std_row['best_epoch']:.1f}"
        )
        formatted_rows.append(row)

    pd.DataFrame(formatted_rows).to_csv(
        output_dir / "summary_mean_std_formatted.csv",
        index=False,
    )


def plot_summary(results: pd.DataFrame, output_dir: Path) -> None:
    grouped = results.groupby("method")[
        ["Dice", "Sensitivity", "Specificity", "Thin_Recall", "clDice"]
    ].mean()

    for metric in grouped.columns:
        figure = plt.figure(figsize=(8, 5))
        grouped[metric].sort_values(ascending=False).plot(kind="bar")
        plt.ylabel(metric)
        plt.title(metric)
        plt.tight_layout()
        figure.savefig(output_dir / f"summary_{metric}.png", dpi=300)
        plt.close(figure)


def run_experiment(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records, val_records, test_records = resolve_dataset_splits(args)

    save_split_manifest(
        output_dir / f"{args.dataset}_split_manifest.json",
        train_records,
        val_records,
        test_records,
    )

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    results = []

    for loss_name in loss_list(args):
        for seed in seeds:
            result = train_single_run(
                args,
                loss_name,
                seed,
                train_records,
                val_records,
                test_records,
            )
            results.append(result)
            pd.DataFrame(results).to_csv(
                output_dir / "all_test_results.csv",
                index=False,
            )

    results_df = pd.DataFrame(results)
    make_summary(results_df, output_dir)
    plot_summary(results_df, output_dir)
