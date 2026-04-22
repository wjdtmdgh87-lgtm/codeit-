"""
tests/models/test_model.py
테스트 FocalLoss 적용 학습 및 추론
YOLOv8n 사전 학습 가중치 로드 + nc=56 헤드 교체
"""

import shutil
import torch
from pathlib import Path
from ultralytics import YOLO
from utils import callbacks as cb
from config import DATASET_YAML, MODELS_DIR, RESULTS_DIR

# ── 테스트 학습 하이퍼파라미터 ────────────────────────
TEST_TRAIN = dict(
    model         = "yolov8m.pt",
    imgsz         = 1024,
    batch         = 8,
    epochs        = 300,
    optimizer     = "AdamW",
    lr0           = 0.01,
    lrf           = 0.01,
    momentum      = 0.937,
    weight_decay  = 5e-4,
    warmup_epochs = 3,
    patience      = 0, # 30, 0=early stopping 해제
    save_period   = 10,
    cls           = 0.5,
    degrees       = 15.0,
    fliplr        = 0.5,
    flipud        = 0.0,
    hsv_h         = 0.015,
    hsv_s         = 0.4,
    hsv_v         = 0.3,
    mosaic        = 1.0,
    mixup         = 0.1,
    copy_paste    = 0.2,
)

def build_model(nc: int = 56) -> YOLO:
    """
    YOLO 모델 가중치를 불러옵니다.
    클래스 개수(nc)에 따른 헤드 교체는 YOLO.train() 시 data.yaml을 읽고 자동으로 수행됩니다.
    """
    model = YOLO(TEST_TRAIN["model"])
    print(f"[모델] {TEST_TRAIN["model"]} 로드 완료")
    return model

def train():
    """학습 실행 후 test_best.pt 를 models/ 에 저장합니다."""
    cfg    = TEST_TRAIN
    device = "0" if torch.cuda.is_available() else "cpu"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_stem = Path(cfg["model"]).stem
    run_name   = f"baseline_{model_stem}"  

    model = build_model(nc=56)
    
    model.add_callback("on_train_start", cb.on_train_start)  # FocalLoss 손실함수(criterion) 변경
    
    model.train(
        data          = str(DATASET_YAML), # dataset.yaml 경로
        project       = str(RESULTS_DIR), # 결과 저장 폴더
        name          = run_name, # 실험 이름 (results/baseline_yolov8n/)
        exist_ok      = True, # 같은 이름 폴더 덮어쓰기 허용
        device        = device, # 학습 장치 (0=GPU, cpu)
        imgsz         = cfg["imgsz"], # 이미지 크기
        batch         = cfg["batch"], # 배치 크기
        epochs        = cfg["epochs"], # 학습 에포크 수
        optimizer     = cfg["optimizer"], # 옵티마이저(SGD)
        lr0           = cfg["lr0"], # 초기 학습률 (0.01)
        lrf           = cfg["lrf"], # 최종 lr 비율 — 최종 lr = lr0 * lrf (0.01*0.01=1e-4)
        momentum      = cfg["momentum"], # SGD 모멘텀 (0.937)
        weight_decay  = cfg["weight_decay"], # 가중치 감쇠 — 과적합 방지 (5e-4)
        warmup_epochs = cfg["warmup_epochs"], # lr 워밍업 epoch 수 (3)
        patience      = cfg["patience"], # early stopping 기준 epoch (0=비활성)
        save_period   = cfg["save_period"], # N epoch마다 체크포인트 저장 (10)
        cls           = cfg["cls"], # 클래스 손실 가중치 (0.5)

        #====== 증강 설정 ========= 
        degrees       = cfg["degrees"], # 회전 데이터 증강 각도 (15.0)
        fliplr        = cfg["fliplr"], # 좌우 반전 데이터 증강 비율 (0.5)
        flipud        = cfg["flipud"], # 상하 반전 데이터 증강 비율 (0.0)
        hsv_h         = cfg["hsv_h"], # Hue jitter (색조 변화)
        hsv_s         = cfg["hsv_s"], # Saturation jitter (채도 변화)
        hsv_v         = cfg["hsv_v"], # Value jitter (밝기 변화)
        mosaic        = cfg["mosaic"], # Mosaic 증강 — 4장 합성 (1.0=항상)
        mixup         = cfg["mixup"], # MixUp 증강 — 두 이미지 혼합 (0.1)
        copy_paste    = cfg["copy_paste"], # Copy-Paste — 희소 클래스 오버샘플 (0.2)
        
        plots         = True,  # PR curve, confusion matrix 등 자동 저장
        verbose       = True, # 학습 로그 상세 출력
    )

    best_src = RESULTS_DIR / run_name / "weights" / "test_best.pt"
    if best_src.exists():
        shutil.copy2(best_src, MODELS_DIR / "test_best_model.pt")
        print(f"[저장] test_best_model.pt → {MODELS_DIR}")

def predict(source: str, weights: str = None, conf: float = 0.25, iou: float = 0.45):
    """학습된 모델로 예측합니다."""
    w     = weights or str(MODELS_DIR / "test_best_model.pt")
    model = YOLO(w)
    
    # 모델 이름을 활용하여 예측 결과 저장 폴더 지정 (예: results/predict_yolo11s)
    model_stem = Path(TEST_TRAIN["model"]).stem
    run_name   = f"predict_{model_stem}"
    
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