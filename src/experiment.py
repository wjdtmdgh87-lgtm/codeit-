"""
[추가 사항]
- YOLO baseline / 비교 실험을 한 파일에서 실행할 수 있도록 새로 만든 파일
- baseline, imgsz_960, longtrain, model_m, strong_aug 실험 설정을 정의함
- 각 실험 실행 후 results.csv 에서 mAP50, mAP50-95, precision, recall 을 추출함
- 추출한 결과를 report_template.py 를 통해 요약 파일로 누적 저장함
"""

from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import TRAIN, RESULTS_DIR
from model import train as run_train
from eval.report_template import save_yolo_result


EXPERIMENTS = {
    "baseline": {},
    "imgsz_960": {
        "imgsz": 960,
    },
    "longtrain": {
        "imgsz": 1024,
        "epochs": 400,
        "save_period": 20,
    },
    "model_m": {
        "model": "yolo11m.pt",
        "imgsz": 1024,
        "batch": 4,
        "epochs": 300,
    },
    "strong_aug": {
        "degrees": 20.0,
        "mosaic": 1.0,
        "mixup": 0.2,
        "copy_paste": 0.3,
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
    experiment_order = [
        "baseline",
        "imgsz_960",
        "longtrain",
        "model_m",
        "strong_aug",
    ]

    for exp_name in experiment_order:
        print(f"\n\n===== {exp_name} 실험 시작 =====")
        try:
            run_experiment(exp_name)
        except Exception as e:
            print(f"[오류] {exp_name} 실험 실패: {e}")

def run_all_experiments():
    experiment_order = [
        "baseline",
        "imgsz_960",
        "longtrain",
        "model_m",
        "strong_aug",
    ]

    for exp_name in experiment_order:
        print(f"\n\n===== {exp_name} 실험 시작 =====")
        try:
            run_experiment(exp_name)
        except Exception as e:
            print(f"[오류] {exp_name} 실험 실패: {e}")