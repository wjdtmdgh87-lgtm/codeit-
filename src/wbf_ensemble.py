"""
src/wbf_ensemble.py
5-Fold WBF (Weighted Boxes Fusion) 앙상블 추론

사용법:
  python main.py --mode wbf --source data/images/test/
  python main.py --mode wbf --source data/images/test/ --conf 0.25

동작:
  1. models/ 폴더에서 fold*_best.pt 파일을 자동으로 수집
  2. 각 모델로 test 이미지 예측
  3. WBF로 5개 예측 결과 합산
  4. 최종 결과 저장 + 클래스별 탐지 수 출력

설치:
  pip install ensemble-boxes
"""

import cv2
import platform
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import ImageFont, ImageDraw, Image

import torch
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

from config import MODELS_DIR, RESULTS_DIR, TRAIN


def get_fold_models() -> list:
    """models/ 폴더에서 fold*_best.pt 파일 목록 반환"""
    models = sorted(MODELS_DIR.glob("fold*_best.pt"))
    if not models:
        raise FileNotFoundError(
            f"[오류] {MODELS_DIR} 에 fold*_best.pt 파일이 없습니다.\n"
            "  먼저 python main.py --mode train_all 을 실행하세요."
        )
    print(f"[앙상블] 모델 {len(models)}개 발견:")
    for m in models:
        print(f"  {m.name}")
    return [str(m) for m in models]


def predict_single(model: YOLO, img_path: str, conf: float, iou: float) -> dict:
    """
    단일 모델로 이미지 예측
    반환: {boxes: [[x1,y1,x2,y2]...], scores: [...], labels: [...]}
    좌표는 0~1 정규화
    """
    results = model.predict(
        img_path,
        conf    = conf,
        iou     = iou,
        verbose = False,
        save    = False,
    )
    r = results[0]
    if len(r.boxes) == 0:
        return {"boxes": [], "scores": [], "labels": []}

    h, w = r.orig_shape
    boxes  = []
    scores = []
    labels = []

    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        boxes.append([x1/w, y1/h, x2/w, y2/h])
        scores.append(float(box.conf[0]))
        labels.append(int(box.cls[0]))

    return {"boxes": boxes, "scores": scores, "labels": labels}


def run_wbf(all_preds: list, iou_thr: float = 0.5, skip_box_thr: float = 0.001) -> dict:
    """
    WBF로 여러 모델의 예측 결과를 합산
    all_preds: [{"boxes":[], "scores":[], "labels":[]}, ...]  모델 수만큼
    """
    boxes_list  = [p["boxes"]  for p in all_preds]
    scores_list = [p["scores"] for p in all_preds]
    labels_list = [p["labels"] for p in all_preds]

    if all(len(b) == 0 for b in boxes_list):
        return {"boxes": [], "scores": [], "labels": []}

    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        iou_thr      = iou_thr,
        skip_box_thr = skip_box_thr,
    )
    return {
        "boxes":  boxes.tolist(),
        "scores": scores.tolist(),
        "labels": [int(l) for l in labels.tolist()],
    }


def get_korean_font(size: int = 28) -> ImageFont.FreeTypeFont:
    """OS별 한글 폰트 자동 탐색"""
    system     = platform.system()
    candidates = []

    if system == "Windows":
        candidates = [
            "C:/Windows/Fonts/malgun.ttf",   # 맑은 고딕
            "C:/Windows/Fonts/gulim.ttc",    # 굴림
        ]
    elif system == "Darwin":                 # macOS
        candidates = [
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/Library/Fonts/AppleGothic.ttf",
        ]
    else:                                    # Linux
        candidates = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf",
        ]

    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)

    print("[경고] 한글 폰트를 찾지 못했습니다. 기본 폰트를 사용합니다.")
    return ImageFont.load_default()


def draw_result(img: np.ndarray, result: dict, class_names: dict, conf_thr: float) -> np.ndarray:
    """예측 결과를 이미지에 그립니다."""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    font    = get_korean_font(size=28)

    h, w = img.shape[:2]
    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        if score < conf_thr:
            continue
        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int(box[2] * w)
        y2 = int(box[3] * h)

        hue       = int((label * 137.5) % 180)
        color     = cv2.cvtColor(np.uint8([[[hue, 220, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
        color_rgb = (int(color[2]), int(color[1]), int(color[0]))

        # 바운딩 박스
        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=2)

        # 텍스트 + 배경
        name = class_names.get(label, f"cls_{label}")
        text = f"{name} {score:.2f}"
        bbox = font.getbbox(text)
        tw   = bbox[2] - bbox[0]
        th   = bbox[3] - bbox[1]
        ty   = max(y1 - th - 10, 0)

        # 텍스트 배경 (불투명)
        draw.rectangle([x1, ty, x1 + tw + 10, ty + th + 10], fill=color_rgb)
        # 텍스트 (흰색)
        draw.text((x1 + 5, ty + 4), text, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def wbf_predict(source: str, conf: float = 0.25, iou: float = 0.45,
                wbf_iou: float = 0.5):
    """
    5-Fold WBF 앙상블 예측 메인 함수

    source  : 이미지 경로 또는 폴더 경로
    conf    : 개별 모델 신뢰도 임계값
    iou     : 개별 모델 NMS IoU
    wbf_iou : WBF 합산 IoU 임계값
    """
    weight_paths = get_fold_models()
    models       = [YOLO(w) for w in weight_paths]
    class_names  = models[0].names

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

    model_stem = Path(TRAIN["model"]).stem
    save_dir   = Path("runs/detect") / f"wbf_{model_stem}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[WBF 앙상블] {len(img_paths)}장 예측 시작")
    print(f"  모델 수   : {len(models)}개")
    print(f"  conf      : {conf}")
    print(f"  wbf_iou   : {wbf_iou}")
    print(f"  저장 위치 : {save_dir}\n")

    class_counts = defaultdict(int)

    for idx, img_path in enumerate(img_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{idx}/{len(img_paths)}] 건너뜀 (손상): {img_path.name}")
            continue

        all_preds = [
            predict_single(m, str(img_path), conf=conf, iou=iou)
            for m in models
        ]
        result = run_wbf(all_preds, iou_thr=wbf_iou)

        for label in result["labels"]:
            class_counts[label] += 1

        img_draw = draw_result(img.copy(), result, class_names, conf_thr=conf)
        cv2.imwrite(str(save_dir / img_path.name), img_draw)

        dets    = [f"{class_names.get(l, l)}({s:.2f})"
                   for l, s in zip(result["labels"], result["scores"])
                   if s >= conf]
        det_str = ", ".join(dets) if dets else "검출 없음"
        print(f"  [{idx}/{len(img_paths)}] {img_path.name} → {len(dets)}개  {det_str}")

    print("\n=== 클래스별 탐지 수 (WBF) ===")
    for cls_id, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {class_names.get(cls_id, cls_id):<45} {count}개")
    print(f"\n  총 탐지       : {sum(class_counts.values())}개")
    print(f"  탐지된 클래스 : {len(class_counts)}종 / {len(class_names)}종")
    print(f"  저장 위치     : {save_dir}")