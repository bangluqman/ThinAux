import argparse

from experiment import run_experiment


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["drive", "stare", "chasedb1"],
        default="drive",
    )

    parser.add_argument(
        "--train-img",
        default="/home/server-ai/LUKMAN PROJECT/FUNDUS/DRIVE/training/images/*",
    )
    parser.add_argument(
        "--train-mask",
        default="/home/server-ai/LUKMAN PROJECT/FUNDUS/DRIVE/training/1st_manual/*",
    )
    parser.add_argument(
        "--test-img",
        default="/home/server-ai/LUKMAN PROJECT/FUNDUS/DRIVE/test/images/*",
    )
    parser.add_argument(
        "--test-mask",
        default="/home/server-ai/LUKMAN PROJECT/FUNDUS/DRIVE/test/1st_manual/*",
    )

    parser.add_argument("--all-img", default="")
    parser.add_argument("--all-mask", default="")
    parser.add_argument("--split-json", default="")
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--test-fraction", type=float, default=0.20)

    parser.add_argument(
        "--experiment",
        choices=["single", "main"],
        default="single",
    )
    parser.add_argument(
        "--loss",
        choices=[
            "bce",
            "bce_dice",
            "weighted_bce",
            "bce_cldice",
            "thinaux_fixed",
            "thinaux_adaptive",
        ],
        default="thinaux_fixed",
    )

    parser.add_argument("--seeds", default="42,123,999")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--train-stride", type=int, default=128)
    parser.add_argument("--inference-stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--fixed-threshold", type=float, default=1.5)
    parser.add_argument("--adaptive-percentile", type=float, default=30.0)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--augment", action="store_true")

    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--cldice-beta", type=float, default=0.5)
    parser.add_argument("--cldice-iterations", type=int, default=10)

    parser.add_argument("--validation-every", type=int, default=1)
    parser.add_argument("--evaluation-thin-threshold", type=float, default=1.5)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default="thinaux_original_results")
    parser.add_argument("--save-predictions", action="store_true")

    args = parser.parse_args()

    if args.dataset in {"stare", "chasedb1"}:
        if not args.all_img or not args.all_mask:
            parser.error("--all-img and --all-mask are required.")

    if args.patch_size % 16 != 0:
        parser.error("--patch-size must be divisible by 16.")

    return args


def main():
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
