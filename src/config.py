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

#기하님 전용: DATASET_YAML = Path("/Users/tnerkfkr/projects/v9/v9_2unzipped/모델러전달_패키지_v9/data.yaml")

# ── 이미지 원본 규격 ───────────────────────────
IMG_W, IMG_H  = 976, 1280

# ── 학습 하이퍼파라미터 ────────────────────────
TRAIN = dict(
    model         = "yolo12n.pt",
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
    mosaic        = 0.8,
    mixup         = 0.05,
    copy_paste    = 0.5,
    close_mosaic = 30, # 마지막 15 epoch는 mosaic 없이 학습
)

# ── K-Fold ─────────────────────────────────────
KFOLD = dict(n_folds=5, use_fold=0, seed=42)

# ── Stage 2 (Crop Classifier) ──────────────────
CROPS_DIR         = DATA_DIR / "crops"
STAGE1_BYPASS_THR = 0.85   # YOLO conf >= 이 값이면 Stage 2 생략

# Stage 2를 신뢰도 무관하게 항상 건너뛸 클래스 ID 목록
STAGE2_SKIP_CLASSES: set = {
    # 예시) 0, 5, 12
    61,  # 쎄로켈정100mg(쿠에티아핀푸마르산염)

}

STAGE2 = dict(
    imgsz    = 300,
    batch    = 64,
    epochs   = 50,
    lr0      = 0.001,
    patience = 10,
)

RARE_CLASS_THR  = 50   # bbox 수 이하 클래스 → crop 증강 대상
CROPS_PER_CLASS = 200  # train 크롭 클래스당 목표 수 (부족하면 증강으로 패딩)