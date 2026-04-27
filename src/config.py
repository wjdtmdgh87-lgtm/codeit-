"""
src/config.py
- 공통 경로 및 학습 하이퍼파라미터 설정 파일
- 운영체제(맥/윈도우)와 상관없이 데이터셋 버전만 지정하면
  모델 코드에서 자동으로 data.yaml을 탐색하도록 구성
"""

from pathlib import Path
import os

# 프로젝트 루트
ROOT = Path(__file__).resolve().parent.parent

# ── 프로젝트 내부 경로 ─────────────────────────────────────────────
DATA_DIR = ROOT / "data"
TRAIN_IMG_DIR = DATA_DIR / "images" / "train"
TEST_IMG_DIR = DATA_DIR / "images" / "test"
ANNOT_DIR = DATA_DIR / "labels" / "train"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# ── 데이터셋 설정 ────────────────────────────────────────────────
# 하드코딩된 절대경로 대신 데이터셋 "버전"만 지정
# 예:
#   맥(zsh): export DATASET_VERSION=v9
#   윈도우(PowerShell): $env:DATASET_VERSION="v9"
# 지정하지 않으면 기본값은 v9
DATASET_VERSION = os.getenv("DATASET_VERSION", "v9")

# 필요하면 data.yaml 절대경로를 환경변수로 직접 지정할 수도 있음
# 예:
#   export DATASET_YAML="/Users/username/projects/v9/.../data.yaml"
# 이 값은 model.py에서 우선적으로 사용함

# ── 이미지 원본 규격 ─────────────────────────────────────────────
IMG_W, IMG_H = 976, 1280

# ── 학습 하이퍼파라미터 ─────────────────────────────────────────
# 현재 baseline 빠른 검증용 설정
TRAIN = dict(
    # 사용할 모델 가중치
    model="yolo11s.pt",

    # 입력 이미지 크기
    imgsz=1280,

    # 배치 크기
    batch=4,

    # 빠른 확인용이므로 5 epoch
    epochs=5,

    # 최적화 설정
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=5e-4,
    warmup_epochs=3,

    # early stopping 비활성화
    patience=0,

    # 몇 epoch마다 checkpoint 저장할지
    save_period=10,

    # loss/augmentation 관련
    cls=1.5,
    degrees=90.0,
    fliplr=0.5,
    flipud=0.5,
    hsv_h=0.015,
    hsv_s=0.2,
    hsv_v=0.2,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.5,

    # 마지막 N epoch는 mosaic 없이 학습
    close_mosaic=15,
)

# ── K-Fold 설정 ────────────────────────────────────────────────
KFOLD = dict(
    n_folds=5,
    use_fold=0,
    seed=42,
)