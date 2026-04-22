"""
[수정 사항]
- 실험 코드에서 재사용할 수 있도록 train() 함수에 run_name, train_overrides 인자를 추가함
- 실험별 설정을 TRAIN 기본값 위에 덮어쓸 수 있게 변경함
- 학습 완료 후 run directory 경로를 반환하도록 변경함
"""

import shutil
import torch
from pathlib import Path
from ultralytics import YOLO
from config import TRAIN, DATASET_YAML, MODELS_DIR, RESULTS_DIR


def build_model(nc: int = 56, model_name: str = None) -> YOLO:
    """
    YOLO 모델 가중치를 불러옵니다.
    클래스 개수(nc)에 따른 헤드 교체는 YOLO.train() 시 data.yaml을 읽고 자동으로 수행됩니다.
    """
    model_path = model_name or TRAIN["model"]
    model = YOLO(model_path)
    print(f"[모델] {model_path} 로드 완료")
    return model


def train(run_name: str = None, train_overrides: dict = None):
    """학습 실행 후 best.pt 를 models/ 에 저장합니다."""
    cfg = TRAIN.copy()
    if train_overrides:
        cfg.update(train_overrides)

    device = "0" if torch.cuda.is_available() else "cpu"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_stem = Path(cfg["model"]).stem
    run_name = run_name or f"baseline_{model_stem}"

    model = build_model(nc=56, model_name=cfg["model"])
    model.train(
        data          = str(DATASET_YAML), # dataset.yaml 경로
        project       = str(RESULTS_DIR), # 결과 저장 폴더
        name          = run_name, # 실험 이름
        exist_ok      = True, # 같은 이름 폴더 덮어쓰기 허용
        device        = device, # 학습 장치 (0=GPU, cpu)
        imgsz         = cfg["imgsz"], # 이미지 크기
        batch         = cfg["batch"], # 배치 크기
        epochs        = cfg["epochs"], # 학습 에포크 수
        optimizer     = cfg["optimizer"], # 옵티마이저
        lr0           = cfg["lr0"], # 초기 학습률
        lrf           = cfg["lrf"], # 최종 lr 비율
        momentum      = cfg["momentum"], # SGD 모멘텀
        weight_decay  = cfg["weight_decay"], # 가중치 감쇠
        warmup_epochs = cfg["warmup_epochs"], # lr 워밍업 epoch 수
        patience      = cfg["patience"], # early stopping 기준 epoch
        save_period   = cfg["save_period"], # N epoch마다 체크포인트 저장
        cls           = cfg["cls"], # 클래스 손실 가중치

        #====== 증강 설정 =========
        degrees       = cfg["degrees"], # 회전 데이터 증강 각도
        fliplr        = cfg["fliplr"], # 좌우 반전 데이터 증강 비율
        flipud        = cfg["flipud"], # 상하 반전 데이터 증강 비율
        hsv_h         = cfg["hsv_h"], # Hue jitter
        hsv_s         = cfg["hsv_s"], # Saturation jitter
        hsv_v         = cfg["hsv_v"], # Value jitter
        mosaic        = cfg["mosaic"], # Mosaic 증강
        mixup         = cfg["mixup"], # MixUp 증강
        copy_paste    = cfg["copy_paste"], # Copy-Paste

        plots         = True,  # PR curve, confusion matrix 등 자동 저장
        verbose       = True, # 학습 로그 상세 출력
    )

    run_dir = RESULTS_DIR / run_name

    best_src = run_dir / "weights" / "best.pt"
    if best_src.exists():
        save_name = f"{run_name}_best.pt"
        shutil.copy2(best_src, MODELS_DIR / save_name)
        print(f"[저장] {save_name} → {MODELS_DIR}")

    return run_dir


def predict(source: str, weights: str = None, conf: float = 0.25, iou: float = 0.45):
    """학습된 모델로 예측합니다."""
    w = weights or str(MODELS_DIR / "best_model.pt")
    model = YOLO(w)

    # 모델 이름을 활용하여 예측 결과 저장 폴더 지정
    model_stem = Path(TRAIN["model"]).stem
    run_name = f"predict_{model_stem}"

    return model.predict(
        source,
        conf=conf,
        iou=iou,
        project=str(Path("runs/detect").absolute()),
        name=run_name,
        exist_ok=False,
        save=True,
        verbose=True
    )


if __name__ == "__main__":
    build_model(nc=56)