"""
테스트 구글 코랩(Colab)
tests/test_config.py

early stopping (조기 중단)
참고: https://wikidocs.net/120091
"""
from pathlib import Path

ROOT = Path("/content")

# ── 경로 ──────────────────────────────────────
DATA_DIR      = ROOT / "data" / "v3"
SPLITS_DIR    = DATA_DIR / "splits"
TRAIN_IMG_DIR = DATA_DIR / "images"
TEST_IMG_DIR  = DATA_DIR / "test_images"
ANNOT_DIR     = DATA_DIR / "labels"
MODELS_DIR    = ROOT / "models"
RESULTS_DIR   = ROOT / "results"

# dataset.yaml 경로 (프로젝트 루트에 위치)
# DATASET_YAML  = ROOT / "data.yaml"
# data.yaml 파일이 ROOT(/content)가 아니라 DATA_DIR(/content/data/v3) 안에 있습니다.
DATASET_YAML  = DATA_DIR / "data.yaml"

# ── 이미지 원본 규격 ───────────────────────────
IMG_W, IMG_H  = 976, 1280

# ── 테스트 학습 하이퍼파라미터 ────────────────────────
TEST_TRAIN = dict(
    model         = "yolov8n.pt",
    imgsz         = 640,
    batch         = 16,
    epochs        = 5,
    optimizer     = "AdamW",
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
)

# ── K-Fold ─────────────────────────────────────
KFOLD = dict(n_folds=5, use_fold=0, seed=42)

# 필요시 참고(2026.04.24 minjae)
# TEST_TRAIN = dict(
#     model         = "yolov8m.pt",
#     imgsz         = 1024,
#     batch         = 8,
#     # epochs        = 300,
#     epochs        = 10,
#     optimizer     = "AdamW",
#     lr0           = 0.01,
#     lrf           = 0.01,
#     momentum      = 0.937,
#     weight_decay  = 5e-4,
#     warmup_epochs = 3,
#     patience      = 0, # 30, 0=early stopping 해제
#     save_period   = 10,
#     cls           = 0.5,
#     degrees       = 15.0,
#     fliplr        = 0.5,
#     flipud        = 0.0,
#     hsv_h         = 0.015,
#     hsv_s         = 0.4,
#     hsv_v         = 0.3,
#     mosaic        = 1.0,
#     mixup         = 0.1,
#     copy_paste    = 0.2,
# )