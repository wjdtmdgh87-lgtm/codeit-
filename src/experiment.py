"""
[수정 사항]
- 모델 1(YOLO11 계열) 빠른 검증용 테스트 5개를 추가함
- 각 테스트는 성능 차이와 학습 안정성을 빠르게 확인하기 위한 짧은 실험용 설정임
- test_01_sanity: 최소 동작 확인용
- test_02_current_short: 현재 모델팀 기본 설정 축약판
- test_03_imgsz_960: 입력 크기 영향 확인용
- test_04_small_model: 더 작은 모델(yolo11n) 비교용
- test_05_no_heavy_aug: 강한 증강 제거 영향 확인용
- 결과는 results.csv 에서 mAP50, mAP50-95, precision, recall 을 추출해 요약 파일로 누적 저장함
"""

from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import TRAIN, RESULTS_DIR
from model import train as run_train
from eval.report_template import save_yolo_result


EXPERIMENTS = {
    "test_01_sanity": {
        "model": "yolo11s.pt",
        "imgsz": 640,
        "batch": 4,
        "epochs": 5,
    },
    "test_02_current_short": {
        "model": "yolo11s.pt",
        "imgsz": 1024,
        "batch": 8,
        "epochs": 5,
    },
    "test_03_imgsz_960": {
        "model": "yolo11s.pt",
        "imgsz": 960,
        "batch": 8,
        "epochs": 5,
    },
    "test_04_small_model": {
        "model": "yolo11n.pt",
        "imgsz": 1024,
        "batch": 8,
        "epochs": 5,
    },
    "test_05_no_heavy_aug": {
        "model": "yolo11s.pt",
        "imgsz": 1024,
        "batch": 8,
        "epochs": 5,
        "degrees": 0.0,
        "fliplr": 0.0,
        "flipud": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
    },
}


def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_experiment_config(exp_name: str) -> dict:
    cfg = deepcopy(TRAIN)
    if exp_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {exp_name}")
    cfg.update(EXPERIMENTS[exp_name])
    return cfg


def extract_metrics_from_results_csv(results_csv: Path) -> dict:
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found: {results_csv}")

    df = pd.read_csv(results_csv)
    last = df.iloc[-1]

    def pick(*names):
        for name in names:
            if name in df.columns:
                return float(last[name])
        return None

    return {
        "precision": pick("metrics/precision(B)", "metrics/precision"),
        "recall": pick("metrics/recall(B)", "metrics/recall"),
        "mAP50": pick("metrics/mAP50(B)", "metrics/mAP50"),
        "mAP50_95": pick("metrics/mAP50-95(B)", "metrics/mAP50-95"),
        "fitness": pick("fitness"),
    }


def run_experiment(exp_name: str):
    cfg = build_experiment_config(exp_name)

    model_stem = Path(cfg["model"]).stem
    run_name = f"{exp_name}_{model_stem}_{now_str()}"

    print(f"\n=== 실험 시작: {run_name} ===")
    print(cfg)

    run_dir = run_train(run_name=run_name, train_overrides=cfg)

    results_csv = run_dir / "results.csv"
    metrics = extract_metrics_from_results_csv(results_csv)

    save_yolo_result(
        experiment_name=run_name,
        config={
            "model": {"weights": cfg["model"]},
            "training": {
                "imgsz": cfg["imgsz"],
                "batch_size": cfg["batch"],
                "epochs": cfg["epochs"],
                "lr0": cfg["lr0"],
            },
        },
        metrics=metrics,
        best_weight_path=str(run_dir / "weights" / "best.pt"),
    )

    print("\n=== 실험 완료 ===")
    print(metrics)


def compare_results():
    summary_path = RESULTS_DIR / "experiment_summary.csv"
    if not summary_path.exists():
        print("experiment_summary.csv 가 없습니다.")
        return

    df = pd.read_csv(summary_path)

    print("\n=== 전체 실험 결과 ===")
    print(df)

    if "mAP50_95" in df.columns:
        print("\n=== mAP50_95 기준 Top 5 ===")
        print(df.sort_values("mAP50_95", ascending=False).head(5))

    if "mAP50" in df.columns:
        print("\n=== mAP50 기준 Top 5 ===")
        print(df.sort_values("mAP50", ascending=False).head(5))


def run_all_experiments():
    """
    모델 1 빠른 비교용 테스트 5개를 순차 실행합니다.
    """
    experiment_order = [
        "test_01_sanity",
        "test_02_current_short",
        "test_03_imgsz_960",
        "test_04_small_model",
        "test_05_no_heavy_aug",
    ]

    for exp_name in experiment_order:
        print(f"\n\n===== {exp_name} 실험 시작 =====")
        try:
            run_experiment(exp_name)
        except Exception as e:
            print(f"[오류] {exp_name} 실험 실패: {e}")