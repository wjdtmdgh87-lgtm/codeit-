"""
src/model.py
- YOLOv8/11 사전 학습 가중치 로드
- 데이터셋 data.yaml 자동 탐색
- split txt 내부 이미지 경로 자동 정규화
- 맥/윈도우 환경에서 경로 하드코딩 의존도를 줄이기 위한 모델 실행 파일
"""

import os
import gc
import shutil
import tempfile
from pathlib import Path

import cv2
import torch
import yaml
from ultralytics import YOLO

from config import TRAIN, MODELS_DIR, RESULTS_DIR, ROOT, DATASET_VERSION


def build_model(nc: int = 56) -> YOLO:
    """
    YOLO 모델 가중치를 불러옵니다.
    클래스 개수(nc)에 따른 헤드 교체는 YOLO.train() 시
    data.yaml의 nc를 읽고 자동으로 수행됩니다.
    """
    model = YOLO(TRAIN["model"])
    print(f"[모델] {TRAIN['model']} 로드 완료")
    return model


def _resolve_dataset_entry(dataset_yaml_path: Path, data: dict, key: str) -> Path | None:
    """
    data.yaml 안 train/val/test 항목을 실제 경로로 해석합니다.
    - 절대경로면 그대로 사용
    - 상대경로면 data.yaml 기준 또는 path 기준으로 해석
    """
    value = data.get(key)
    if not value:
        return None

    dataset_root = dataset_yaml_path.parent

    raw_path = data.get("path")
    if raw_path:
        rp = Path(str(raw_path))
        if rp.is_absolute():
            dataset_root = rp
        else:
            dataset_root = (dataset_yaml_path.parent / rp).resolve()

    p = Path(str(value))
    if p.is_absolute():
        return p.resolve()

    return (dataset_root / p).resolve()


def _score_dataset_yaml(candidate_yaml: Path, version_lower: str) -> tuple[int, bool]:
    """
    data.yaml 후보 점수 계산.
    - 실제 train/val/test 경로 존재 여부를 검사
    - usable하지 않으면 선택 대상에서 제외
    """
    try:
        with open(candidate_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return (-9999, False)

    score = 0
    full = str(candidate_yaml)
    sp = full.lower()

    # version 문자열이 경로에 포함되면 가산점
    if version_lower in sp:
        score += 20

    parent = candidate_yaml.parent
    if (parent / "images").exists():
        score += 10
    if (parent / "splits").exists():
        score += 10
    if (parent / "test_images").exists():
        score += 10

    # 더 구체적인 패키지 구조 우대
    if "desktop" in sp:
        score += 30
    if "모델러_전달_패키지" in full:
        score += 25
    if "unzipped" in sp:
        score += 10
    if "v9_2" in sp:
        score += 20

    train_path = _resolve_dataset_entry(candidate_yaml, data, "train")
    val_path = _resolve_dataset_entry(candidate_yaml, data, "val")
    test_path = _resolve_dataset_entry(candidate_yaml, data, "test")

    usable = True

    if train_path is None or not train_path.exists():
        usable = False
    if val_path is None or not val_path.exists():
        usable = False

    if test_path is not None and test_path.exists():
        score += 5

    if train_path is not None and train_path.suffix == ".txt" and train_path.exists():
        score += 15
    if val_path is not None and val_path.suffix == ".txt" and val_path.exists():
        score += 15

    return (score, usable)


def find_dataset_yaml(version: str) -> Path:
    """
    맥/윈도우 공통으로 data.yaml 자동 탐색.

    우선순위:
    1) 환경변수 DATASET_YAML
    2) Desktop / projects / Downloads / home 아래의 모든 data.yaml 탐색
       - 실제 train/val이 존재하는 usable 후보만 채택
       - version 문자열, Desktop, 모델러_전달_패키지, unzipped, v9_2 등에 가산점
    3) 마지막 fallback으로 프로젝트 내부 data.yaml
    """
    env_yaml = os.getenv("DATASET_YAML")
    if env_yaml:
        p = Path(env_yaml).expanduser().resolve()
        if p.exists():
            print(f"[DATASET] env override: {p}")
            return p

    search_roots = [
        Path.home() / "Desktop",
        Path.home() / "projects",
        Path.home() / "Downloads",
        Path.home(),
    ]

    version_lower = version.lower()
    candidates = []

    for search_root in search_roots:
        if not search_root.exists():
            continue

        for p in search_root.rglob("data.yaml"):
            # 현재 experiment-model 내부 data.yaml은 자동 탐색 후보에서 제외
            if p.resolve() == (ROOT / "data.yaml").resolve():
                continue

            score, usable = _score_dataset_yaml(p.resolve(), version_lower)
            if usable:
                candidates.append((score, p.resolve()))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)

        print("[DATASET] candidate ranking:")
        for score, path in candidates[:5]:
            print(f"  score={score:>3}  path={path}")

        best = candidates[0][1]
        print(f"[DATASET] auto-detected: {best}")
        return best

    local_yaml = ROOT / "data.yaml"
    if local_yaml.exists():
        print(f"[DATASET] local fallback: {local_yaml}")
        return local_yaml.resolve()

    raise FileNotFoundError(
        f"Could not find usable data.yaml for dataset version '{version}'."
    )


def _normalize_split_file(split_file: Path, dataset_root: Path) -> Path:
    """
    split txt 안 경로를 절대경로로 정규화한 임시 txt 생성

    허용 입력 예:
    - /absolute/path/to/image.jpg
    - images/example.jpg
    - ./images/example.jpg
    - example.jpg
    """
    if not split_file.exists():
        raise FileNotFoundError(f"split file not found: {split_file}")

    lines = split_file.read_text(encoding="utf-8").splitlines()
    fixed = []

    for raw in lines:
        s = raw.strip()
        if not s:
            continue

        # 윈도우 백슬래시 대비
        s = s.replace("\\", "/")
        p = Path(s)

        # 이미 절대경로면 그대로 유지
        if p.is_absolute():
            fixed.append(str(p))
            continue

        # ./ 제거
        if s.startswith("./"):
            s = s[2:]

        # images/... 형태
        if "/images/" in s:
            filename = s.split("/images/")[-1]
            fixed.append(str((dataset_root / "images" / filename).resolve()))
            continue

        if s.startswith("images/"):
            filename = s.replace("images/", "", 1)
            fixed.append(str((dataset_root / "images" / filename).resolve()))
            continue

        if s.startswith("test_images/"):
            filename = s.replace("test_images/", "", 1)
            fixed.append(str((dataset_root / "test_images" / filename).resolve()))
            continue

        # 파일명만 들어있으면 images 밑으로 가정
        fixed.append(str((dataset_root / "images" / s).resolve()))

    tmp_dir = Path(tempfile.mkdtemp(prefix="normalized_split_"))
    out_path = tmp_dir / split_file.name
    out_path.write_text("\n".join(fixed) + "\n", encoding="utf-8")
    return out_path


def prepare_dataset_yaml(dataset_yaml_path: Path) -> Path:
    """
    data.yaml을 현재 OS에서 안전하게 동작하는 임시 normalized yaml로 변환합니다.

    변환 내용:
    - train / val / test를 절대경로로 정규화
    - split txt 내부 이미지 경로도 절대경로로 정규화
    - path는 '.'로 무력화하여 실행 위치 의존성 제거
    """
    dataset_yaml_path = dataset_yaml_path.expanduser().resolve()

    if not dataset_yaml_path.exists():
        raise FileNotFoundError(f"dataset yaml not found: {dataset_yaml_path}")

    with open(dataset_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    dataset_root = dataset_yaml_path.parent

    raw_path = data.get("path")
    if raw_path:
        rp = Path(str(raw_path))
        if rp.is_absolute():
            dataset_root = rp
        else:
            dataset_root = (dataset_yaml_path.parent / rp).resolve()

    def normalize_entry(value: str) -> str:
        s = str(value).replace("\\", "/")
        p = Path(s)

        if p.is_absolute():
            return str(p)

        if s.endswith(".txt"):
            split_path = (dataset_root / s).resolve()
            return str(_normalize_split_file(split_path, dataset_root))

        return str((dataset_root / s).resolve())

    if "train" in data:
        data["train"] = normalize_entry(data["train"])
    if "val" in data:
        data["val"] = normalize_entry(data["val"])
    if "test" in data:
        data["test"] = normalize_entry(data["test"])

    # 실행 위치 의존성 제거
    data["path"] = "."

    tmp_dir = Path(tempfile.mkdtemp(prefix="normalized_yaml_"))
    out_yaml = tmp_dir / "data.normalized.yaml"

    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    print(f"[DATASET] normalized yaml: {out_yaml}")
    print(f"[DATASET] train: {data.get('train')}")
    print(f"[DATASET] val: {data.get('val')}")
    print(f"[DATASET] test: {data.get('test')}")

    return out_yaml


def _run_train(yaml_path: str, run_name: str):
    """
    학습 실행 공통 함수
    """
    cfg = TRAIN

    # 1. CUDA(NVIDIA GPU)
    if torch.cuda.is_available():
        device = "0"
    # 2. MPS(Apple Silicon GPU)
    elif torch.backends.mps.is_available():
        device = "mps"
    # 3. CPU
    else:
        device = "cpu"

    print(f"현재 사용 중인 디바이스: {device}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model(nc=56)
    model.train(
        data=yaml_path,
        project=str(RESULTS_DIR),
        name=run_name,
        exist_ok=True,
        device=device,
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        epochs=cfg["epochs"],
        optimizer=cfg["optimizer"],
        lr0=cfg["lr0"],
        lrf=cfg["lrf"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
        warmup_epochs=cfg["warmup_epochs"],
        patience=cfg["patience"],
        save_period=cfg["save_period"],
        cls=cfg["cls"],
        degrees=cfg["degrees"],
        fliplr=cfg["fliplr"],
        flipud=cfg["flipud"],
        hsv_h=cfg["hsv_h"],
        hsv_s=cfg["hsv_s"],
        hsv_v=cfg["hsv_v"],
        mosaic=cfg["mosaic"],
        mixup=cfg["mixup"],
        copy_paste=cfg["copy_paste"],
        close_mosaic=cfg["close_mosaic"],
        plots=True,
        verbose=True,
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
    """
    data.yaml 기준 단일 baseline 학습
    """
    cfg = TRAIN
    model_stem = Path(cfg["model"]).stem
    run_name = f"baseline_{model_stem}"

    detected_yaml = find_dataset_yaml(DATASET_VERSION)
    normalized_yaml = prepare_dataset_yaml(detected_yaml)

    _run_train(str(normalized_yaml), run_name)

    best_src = RESULTS_DIR / run_name / "weights" / "best.pt"
    if best_src.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_src, MODELS_DIR / "best_model.pt")
        print(f"[저장] best_model.pt → {MODELS_DIR}")


def train_all_folds(n_folds: int = 5):
    """
    fold 1~n 을 순서대로 자동 학습합니다.
    현재는 기본 local data.yaml 기반 임시 구현입니다.
    필요 시 이 부분도 자동 탐색 버전에 맞게 추가 정리할 수 있습니다.
    """
    import yaml

    model_stem = Path(TRAIN["model"]).stem
    base_yaml = ROOT / "data.yaml"

    print(f"=== {n_folds}-Fold 전체 학습 시작 ({model_stem}) ===\n")

    for fold in range(1, n_folds + 1):
        print(f"\n{'=' * 50}")
        print(f"  Fold {fold} / {n_folds} 학습 시작")
        print(f"{'=' * 50}")

        with open(base_yaml, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        data["train"] = f"data/splits/fold{fold}_train_oversampled.txt"
        data["val"] = f"data/splits/fold{fold}_val.txt"

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
    """
    학습된 모델로 예측합니다.
    """
    w = weights or str(MODELS_DIR / "best_model.pt")
    model = YOLO(w)

    model_stem = Path(TRAIN["model"]).stem
    run_name = f"predict_{model_stem}"

    results = model.predict(
        source,
        conf=conf,
        iou=iou,
        agnostic_nms=True,
        project=str(Path("runs/detect").absolute()),
        name=run_name,
        exist_ok=False,
        save=False,
        verbose=True,
    )

    print_mapping = {}
    if use_ocr:
        try:
            from ocr_correction import build_print_mapping, correct_predictions
            print_mapping = build_print_mapping()
        except ImportError:
            print("[경고] ocr_correction 모듈을 찾을 수 없습니다. OCR 보정을 건너뜁니다.")
            use_ocr = False

    if results and hasattr(results[0], "save_dir"):
        save_dir = Path(results[0].save_dir)
    else:
        save_dir = Path("runs/detect") / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    from collections import defaultdict
    class_counts = defaultdict(int)
    class_names = model.names

    for r in results:
        img_path = Path(r.path)
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        result = {
            "boxes": [box.xyxy[0].tolist() for box in r.boxes],
            "scores": [float(box.conf[0]) for box in r.boxes],
            "labels": [int(box.cls[0]) for box in r.boxes],
        }

        h, w = r.orig_shape
        result["boxes"] = [[x1 / w, y1 / h, x2 / w, y2 / h] for x1, y1, x2, y2 in result["boxes"]]

        if use_ocr and print_mapping:
            result = correct_predictions(img, result, class_names, print_mapping, conf_thr=conf)

        for label in result["labels"]:
            class_counts[label] += 1

        from wbf_ensemble import draw_result
        img_draw = draw_result(img.copy(), result, class_names, conf_thr=conf)
        cv2.imwrite(str(save_dir / img_path.name), img_draw)

    print("\n=== 클래스별 탐지 수 ===")
    for cls_id, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {class_names[cls_id]:<45} {count}개")
    print(f"\n  총 탐지: {sum(class_counts.values())}개")
    print(f"  탐지된 클래스: {len(class_counts)}종 / {len(class_names)}종")
    print(f"  저장 위치: {save_dir}")

    return results


if __name__ == "__main__":
    build_model(nc=56)