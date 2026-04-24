"""
[수정 사항]
- 실험 설정을 EXPERIMENTS 딕셔너리 하드코딩 방식에서 YAML 파일 로드 방식으로 변경함
- config/experiments/*.yaml 파일을 읽어 실험 설정을 구성함
- experiment_name 필드를 YAML에서 읽고, 없으면 파일명(stem)을 사용함
- 각 실험 완료 후 results.csv 에서 mAP50, mAP50-95, precision, recall 을 추출함
- 추출한 결과를 report_template.py 를 통해 요약 파일로 누적 저장함
"""

from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from config import TRAIN, RESULTS_DIR, ROOT
from model import train as run_train
from eval.report_template import save_yolo_result


EXPERIMENTS_DIR = ROOT / "config" / "experiments"


def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_experiment_yaml(exp_name: str) -> dict:
    """
    exp_name에 대응하는 YAML 파일을 찾아 로드합니다.
    예:
      exp_name = "yolo_mac_baseline"
      -> config/experiments/yolo_mac_baseline.yaml
    """
    yaml_path = EXPERIMENTS_DIR / f"{exp_name}.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"Experiment yaml not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return data


def build_experiment_config(exp_name: str) -> dict:
    """
    TRAIN 기본값 위에 YAML 실험 설정을 덮어씁니다.
    """
    cfg = deepcopy(TRAIN)
    yaml_cfg = load_experiment_yaml(exp_name)

    # experiment_name은 실행 설정이 아니라 메타데이터이므로 제거
    yaml_cfg.pop("experiment_name", None)

    cfg.update(yaml_cfg)
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
    yaml_cfg = load_experiment_yaml(exp_name)
    cfg = build_experiment_config(exp_name)

    yaml_experiment_name = yaml_cfg.get("experiment_name", exp_name)
    model_stem = Path(cfg["model"]).stem
    run_name = f"{yaml_experiment_name}_{model_stem}_{now_str()}"

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
                "patience": cfg.get("patience"),
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


def list_experiments() -> list[str]:
    """
    config/experiments 폴더 안 yaml 파일 목록을 반환합니다.
    """
    if not EXPERIMENTS_DIR.exists():
        return []

    return sorted([p.stem for p in EXPERIMENTS_DIR.glob("*.yaml")])


def run_all_experiments():
    """
    config/experiments 폴더 안의 모든 YAML 실험을 순차 실행합니다.
    """
    experiment_order = list_experiments()

    if not experiment_order:
        print("실행할 experiment yaml 이 없습니다.")
        return

    for exp_name in experiment_order:
        print(f"\n\n===== {exp_name} 실험 시작 =====")
        try:
            run_experiment(exp_name)
        except Exception as e:
            print(f"[오류] {exp_name} 실험 실패: {e}")