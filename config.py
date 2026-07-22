import os

BASE_PATH = "DRIVE"
TRAIN_IMG = os.path.join(BASE_PATH, "training", "images", "*")
TRAIN_MASK = os.path.join(BASE_PATH, "training", "1st_manual", "*")
TEST_IMG = os.path.join(BASE_PATH, "test", "images", "*")
TEST_MASK = os.path.join(BASE_PATH, "test", "1st_manual", "*")

PATCH_SIZE = 128
STRIDE = 64
BATCH_SIZE = 12
LR = 1e-4
EPOCHS = 200
ALPHA = 0.4
THIN_THRESHOLD = 1.5
ADAPTIVE_PERCENTILE = 30
PRED_THRESHOLD = 0.5
SEEDS = [42, 123, 999]
VAL_RATIO = 0.2
SPLIT_SEED = 2026
DEVICE = "cuda"
NUM_WORKERS = 0
OUTPUT_DIR = "result_thinaux"
