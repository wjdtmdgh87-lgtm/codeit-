"""
src/config.py
"""

from pathlib import Path

ROOT          = Path(__file__).resolve().parent.parent

# ── 경로 ──────────────────────────────────────
DATA_DIR      = ROOT / "data"
TRAIN_IMG_DIR = DATA_DIR / "images" / "train"
TEST_IMG_DIR  = DATA_DIR / "images" / "test"
ANNOT_DIR     = DATA_DIR / "labels" / "train"
MODELS_DIR    = ROOT / "models"
RESULTS_DIR   = ROOT / "results"

# dataset.yaml 경로 (프로젝트 루트에 위치)
DATASET_YAML  = ROOT / "data.yaml"

# ── 이미지 원본 규격 ───────────────────────────
IMG_W, IMG_H  = 976, 1280

# ── 학습 하이퍼파라미터 ────────────────────────
TRAIN = dict(
    model         = "yolo11s.pt",
    imgsz         = 1280,
    batch         = 4,
    epochs        = 150,
    optimizer     = "SGD",
    lr0           = 0.01,
    lrf           = 0.01,
    momentum      = 0.937,
    weight_decay  = 5e-4,
    warmup_epochs = 3,
    patience      = 30, # 30, 0=early stopping 해제
    save_period   = 10,
    cls           = 1.5,
    degrees       = 90.0,
    fliplr        = 0.5,
    flipud        = 0.5,
    hsv_h         = 0.015,
    hsv_s         = 0.2,
    hsv_v         = 0.2,
    mosaic        = 1.0,
    mixup         = 0.1,
    copy_paste    = 0.5,
    close_mosaic = 15, # 마지막 15 epoch는 mosaic 없이 학습
)

# ── K-Fold ─────────────────────────────────────
KFOLD = dict(n_folds=5, use_fold=0, seed=42)