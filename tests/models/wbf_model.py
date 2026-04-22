"""
tests/models/wbf_model.py
테스트 5-Fold 데이터 분할 및 WBF 앙상블 추론
YOLOv8n 사전 학습 가중치 로드 + nc=56 헤드 교체
"""

import SafeCustomFocalLoss  # Focal Loss
import shutil
import torch
from pathlib import Path
from ultralytics import YOLO
from config import TRAIN, DATASET_YAML, MODELS_DIR, RESULTS_DIR

def build_model(nc: int = 56, model_name: str = "yolov8m.pt") -> YOLO:
    """
    YOLO 모델 가중치를 불러옵니다.
    클래스 개수(nc)에 따른 헤드 교체는 YOLO.train() 시 data.yaml을 읽고 자동으로 수행됩니다.
    """
    model = YOLO(model_name)
    print(f"[모델] {model_name} 로드 완료")
    return model

def train():
    """학습 실행 후 best.pt 를 models/ 에 저장합니다."""
    # TODO: yolov8m 모델 학습 로직 추후 구현 예정(2026.04.22 minjae)
    # pass