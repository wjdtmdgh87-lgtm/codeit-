"""
[수정 사항]
- config.settings import 충돌 문제 해결
- 기존 src/config.py 의 RESULTS_DIR 을 기준으로 실험 결과 저장
- YOLO 실험 결과를 json / csv 형식으로 누적 저장
"""

from pathlib import Path
from utils.io_utils import save_json, append_csv_row
from config import RESULTS_DIR


def save_yolo_result(experiment_name: str, config: dict, metrics: dict, best_weight_path: str = ""):
    yolo_output_dir = RESULTS_DIR

    report = {
        "experiment_name": experiment_name,
        "config": config,
        "metrics": metrics,
        "best_weight_path": best_weight_path,
    }

    save_json(report, yolo_output_dir / f"{experiment_name}.json")

    summary_row = {
        "experiment_name": experiment_name,
        "model": config["model"]["weights"],
        "imgsz": config["training"]["imgsz"],
        "epochs": config["training"]["epochs"],
        "batch_size": config["training"]["batch_size"],
        "lr0": config["training"]["lr0"],
        "mAP50": metrics.get("mAP50"),
        "mAP50_95": metrics.get("mAP50_95"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "fitness": metrics.get("fitness"),
        "best_weight_path": best_weight_path,
    }

    append_csv_row(yolo_output_dir / "experiment_summary.csv", summary_row)