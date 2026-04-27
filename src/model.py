"""
src/model.py
YOLOv8/11 사전 학습 가중치 로드 + nc=56 헤드 교체
"""

import gc
import shutil
import torch
import cv2
from pathlib import Path
from ultralytics import YOLO
from config import TRAIN, DATASET_YAML, MODELS_DIR, RESULTS_DIR, ROOT


def build_model(nc: int = 56) -> YOLO:
    """
    YOLO 모델 가중치를 불러옵니다.
    클래스 개수(nc)에 따른 헤드 교체는 YOLO.train() 시 data.yaml을 읽고 자동으로 수행됩니다.
    """
    model = YOLO(TRAIN["model"])
    print(f"[모델] {TRAIN['model']} 로드 완료")
    return model


def _run_train(yaml_path: str, run_name: str):
    """학습 실행 공통 함수"""
    cfg = TRAIN

    if torch.cuda.is_available():
        device = "0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"현재 사용 중인 디바이스: {device}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model(nc=56)
    model.train(
        data          = yaml_path,
        project       = str(RESULTS_DIR),
        name          = run_name,
        exist_ok      = True,
        device        = device,
        imgsz         = cfg["imgsz"],
        batch         = cfg["batch"],
        epochs        = cfg["epochs"],
        optimizer     = cfg["optimizer"],
        lr0           = cfg["lr0"],
        lrf           = cfg["lrf"],
        momentum      = cfg["momentum"],
        weight_decay  = cfg["weight_decay"],
        warmup_epochs = cfg["warmup_epochs"],
        patience      = cfg["patience"],
        save_period   = cfg["save_period"],
        cls           = cfg["cls"],
        degrees       = cfg["degrees"],
        fliplr        = cfg["fliplr"],
        flipud        = cfg["flipud"],
        hsv_h         = cfg["hsv_h"],
        hsv_s         = cfg["hsv_s"],
        hsv_v         = cfg["hsv_v"],
        mosaic        = cfg["mosaic"],
        mixup         = cfg["mixup"],
        copy_paste    = cfg["copy_paste"],
        close_mosaic  = cfg["close_mosaic"],
        plots         = True,
        verbose       = True,
    )

    best_src = RESULTS_DIR / run_name / "weights" / "best.pt"
    if best_src.exists():
        dst = MODELS_DIR / f"{run_name}_best.pt"
        shutil.copy2(best_src, dst)
        print(f"[저장] {dst.name} → {MODELS_DIR}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


def train():
    """data.yaml 기준 단일 학습 (fold 1 기본)"""
    cfg        = TRAIN
    model_stem = Path(cfg["model"]).stem
    run_name   = f"baseline_{model_stem}"
    _run_train(str(DATASET_YAML), run_name)

    best_src = RESULTS_DIR / run_name / "weights" / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, MODELS_DIR / "best_model.pt")
        print(f"[저장] best_model.pt → {MODELS_DIR}")


def train_all_folds(n_folds: int = 5):
    """
    fold 1~n 을 순서대로 자동 학습합니다.
    각 fold 결과 → models/fold{N}_{model}_best.pt
    WBF 앙상블 시 이 파일들을 사용합니다.
    """
    import yaml

    model_stem = Path(TRAIN["model"]).stem
    base_yaml  = ROOT / "data.yaml"

    print(f"=== {n_folds}-Fold 전체 학습 시작 ({model_stem}) ===\n")

    for fold in range(1, n_folds + 1):
        print(f"\n{'='*50}")
        print(f"  Fold {fold} / {n_folds} 학습 시작")
        print(f"{'='*50}")

        with open(base_yaml, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        data["train"] = f"data/splits/fold{fold}_train_oversampled.txt"
        data["val"]   = f"data/splits/fold{fold}_val.txt"
        tmp_yaml = ROOT / f"data_fold{fold}.yaml"
        with open(tmp_yaml, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

        run_name = f"fold{fold}_{model_stem}"
        _run_train(str(tmp_yaml), run_name)

        print(f"  Fold {fold} 완료\n")

    print("\n=== 전체 Fold 학습 완료 ===")
    print("저장된 모델:")
    for f in sorted(MODELS_DIR.glob("fold*_best.pt")):
        print(f"  {f.name}")


def predict(source: str, weights: str = None, conf: float = 0.25, iou: float = 0.45, use_ocr: bool = True):
    """학습된 모델로 예측합니다."""
    w     = weights or str(MODELS_DIR / "best_model.pt")
    model = YOLO(w)

    model_stem = Path(TRAIN["model"]).stem
    run_name   = f"predict_{model_stem}"

    # 이미지 목록 수집 (정렬로 순서 보장)
    src = Path(source)
    if src.is_dir():
        img_paths = sorted(
            list(src.glob("*.png")) +
            list(src.glob("*.jpg")) +
            list(src.glob("*.jpeg"))
        )
    else:
        img_paths = [src]

    if not img_paths:
        print(f"[오류] 이미지 없음: {source}")
        return

    # predict에 같은 목록 전달 → results 순서 일치 보장
    results = model.predict(
        img_paths,
        conf     = conf,
        iou      = iou,
        save     = False,
        verbose  = True,
    )

    # OCR 매핑 테이블 로드 (use_ocr=False면 빈 dict)
    print_mapping = {}
    if use_ocr:
        from ocr_correction import build_print_mapping, correct_predictions
        print_mapping = build_print_mapping()

    # 저장 폴더
    save_dir = Path("runs/detect") / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    from collections import defaultdict
    class_counts = defaultdict(int)
    class_names  = model.names

    for r, img_path in zip(results, img_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  건너뜀 (손상): {img_path.name}")
            continue

        result = {
            "boxes":  [box.xyxy[0].tolist() for box in r.boxes],
            "scores": [float(box.conf[0]) for box in r.boxes],
            "labels": [int(box.cls[0]) for box in r.boxes],
        }

        # 좌표 정규화 (0~1)
        h, w = r.orig_shape
        result["boxes"] = [[x1/w, y1/h, x2/w, y2/h] for x1, y1, x2, y2 in result["boxes"]]

        # OCR 보정
        if use_ocr and print_mapping:
            result = correct_predictions(img, result, class_names, print_mapping, conf_thr=conf)

        for label in result["labels"]:
            class_counts[label] += 1

        # 이미지 저장
        from wbf_ensemble import draw_result
        img_draw = draw_result(img.copy(), result, class_names, conf_thr=conf)
        cv2.imwrite(str(save_dir / img_path.name), img_draw)

    print("\n=== 클래스별 탐지 수 ===")
    for cls_id, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {class_names[cls_id]:<45} {count}개")
    print(f"\n  총 탐지: {sum(class_counts.values())}개")
    print(f"  탐지된 클래스: {len(class_counts)}종 / {len(class_names)}종")

    return results


if __name__ == "__main__":
    build_model(nc=56)