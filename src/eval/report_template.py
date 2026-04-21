from src.utils.io_utils import save_json, append_csv_row
from config.settings import YOLO_OUTPUT_DIR


def save_yolo_result(experiment_name: str, config: dict, metrics: dict, best_weight_path: str = ""):
    report = {
        "experiment_name": experiment_name,
        "config": config,
        "metrics": metrics,
        "best_weight_path": best_weight_path,
    }

    save_json(report, YOLO_OUTPUT_DIR / f"{experiment_name}.json")

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
        "best_weight_path": best_weight_path,
    }

    append_csv_row(YOLO_OUTPUT_DIR / "yolo_experiment_summary.csv", summary_row)