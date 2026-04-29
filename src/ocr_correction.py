"""
src/ocr_correction.py
EasyOCR 기반 각인(인쇄문자) 보정 모듈

동작:
  1. code_mapping.csv에서 클래스별 각인(print_front, print_back) 로드
  2. YOLO 신뢰도 < OCR_TRIGGER_CONF 인 박스에만 OCR 실행
  3. OCR 텍스트가 매핑 테이블의 클래스와 일치하면 해당 클래스로 확정 + score 상향
  4. 매핑 실패 시 원본 YOLO 결과 유지

설치:
  pip install easyocr
"""

import re
import csv
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch

from config import ROOT

OCR_TRIGGER_CONF = 0.85  # 이 미만 박스만 OCR 적용 (Stage 2 bypass 임계값과 동일)
CROP_PAD         = 0.15  # bbox 크롭 시 가로·세로 패딩 비율
ROTATION_ANGLES  = [90, 180, 270]  # 각인 방향 탐색 (0°는 EasyOCR 기본)
MIN_CROP_PX      = 128   # 업스케일 보장 최소 변 길이

# 자주 헷갈리는 알약 클래스 ID 목록 (신뢰도 무관하게 무조건 OCR 실행)
CONFUSING_CLASSES = {

    7,   # 에어탈정(아세클로페낙)
    11,  # 써스펜8시간이알서방정 650mg
    14,  # 크레스토정 20mg
    23,  # 노바스크정 5mg
    27,  # 아토르바정 10mg
    30,  # 자누비아정 50mg
    40,  # 트라젠타정(리나글립틴)
    44,  # 트라젠타듀오정 2.5/850mg
    45,  # 아질렉트정(라사길린메실산염)
    51,  # 제미메트서방정 50/1000mg
    53,  # 로수젯정10/5밀리그램
}

_ocr_reader = None  # 전역 싱글턴 — 최초 1회만 로드


def _get_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        use_gpu = torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        print("[OCR] EasyOCR 초기화 중 (처음 실행 시 모델 다운로드가 있을 수 있습니다)...")
        _ocr_reader = easyocr.Reader(["en"], gpu=use_gpu)
        print("[OCR] 초기화 완료")
    return _ocr_reader


def _normalize(text: str) -> str:
    """각인 텍스트 정규화: 분할선 설명자 제거 → 대문자 → 영숫자만 유지"""
    text = str(text)
    text = re.sub(r"십자분할선|분할선|십자", "", text)
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def build_print_mapping(csv_path=None) -> dict:
    """
    code_mapping.csv에서 정규화된 각인 텍스트 → [class_idx, ...] 매핑 생성.

    Returns:
        {normalized_inscription: [class_idx, ...]}
    """
    csv_path = Path(csv_path) if csv_path else ROOT / "code_mapping.csv"
    mapping = defaultdict(list)

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["class_idx"])
            for field in ("print_front", "print_back"):
                raw = row.get(field, "").strip()
                if not raw:
                    continue
                norm = _normalize(raw)
                if len(norm) >= 2:  # 너무 짧은 각인은 노이즈로 간주
                    mapping[norm].append(idx)

    print(f"[OCR] 각인 매핑 로드: {len(mapping)}개 고유 각인")
    return dict(mapping)


def _preprocess_for_ocr(img_crop):
    """
    OCR 인식률 향상을 위한 4가지 전처리 버전 생성
    """
    # 그레이스케일 변환
    if len(img_crop.shape) == 3:
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_crop.copy()

    # 최소 크기(MIN_CROP_PX) 미만일 경우 업스케일링
    h, w = gray.shape
    if min(h, w) < MIN_CROP_PX:
        scale = MIN_CROP_PX / min(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    clahe_wide  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    clahe_tight = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, 2))  # 타이트 타일로 국소 대비 강화

    def unsharp(img, sigma=2.0, strength=1.8):
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        return cv2.addWeighted(img, strength, blur, -(strength - 1), 0)

    # v1: 표준 음각 (CLAHE + Unsharp)
    v1 = unsharp(clahe_wide.apply(gray))

    # v2: 양각 및 반전 대비 (Invert + CLAHE + Unsharp)
    v2 = unsharp(clahe_wide.apply(cv2.bitwise_not(gray)))

    # v3: 노이즈 제거 (Bilateral Filter + CLAHE)
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    v3 = clahe_wide.apply(denoised)

    # v4: 타이트 CLAHE — 각인 부위의 매우 국소적인 명암 차이 강조
    v4 = unsharp(clahe_tight.apply(gray))

    return [v1, v2, v3, v4]


def read_text_from_crop(img_crop) -> list:
    """
    크롭 이미지에서 EasyOCR로 텍스트 인식 (다중 전처리 변형 적용).

    EasyOCR 내부의 rotation_info를 사용하면 세로로 된 'SK'를 가로 'U'로 잘못 인식할 때
    'U'의 신뢰도가 더 높으면 'SK'를 버리는 문제가 있습니다.
    따라서 내부 회전을 끄고, 4방향으로 직접 회전시킨 이미지를 모두 OCR에 넣어
    나오는 모든 텍스트(오인식 포함)를 수집한 뒤 매핑 DB로 걸러냅니다.
    """
    # 4가지 방향 물리적 회전
    crops = [
        img_crop,
        cv2.rotate(img_crop, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(img_crop, cv2.ROTATE_180),
        cv2.rotate(img_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]

    reader = _get_reader()
    all_results: list = []
    seen: set = set()

    for crop in crops:
        variants = _preprocess_for_ocr(crop)
        for variant in variants:
            try:
                # rotation_info 파라미터 제거 (내부 회전 비교 방지)
                raw = reader.readtext(
                    variant,
                    detail=1,
                    mag_ratio=2.0,
                    contrast_ths=0.1,
                    adjust_contrast=0.8,
                    text_threshold=0.5,
                    low_text=0.3,
                )
            except Exception:
                continue

            if not raw:
                continue

            # 1. 개별 토큰 수집
            for (_, text, conf) in raw:
                t = str(text).strip()
                if t and t not in seen:
                    seen.add(t)
                    all_results.append((t, float(conf)))

            # 2. 쪼개진 박스 합치기 (예: 'S' 와 'K' 가 분리된 경우 대비)
            if len(raw) > 1:
                combined_text = "".join([str(text).strip() for _, text, _ in raw])
                avg_conf = sum([float(conf) for _, _, conf in raw]) / len(raw)
                if combined_text and combined_text not in seen:
                    seen.add(combined_text)
                    all_results.append((combined_text, float(avg_conf)))

    return all_results


def _find_match(ocr_norm: str, mapping: dict, yolo_label: int = None):
    """
    정규화된 OCR 문자열로 매핑을 탐색해 class_idx 반환.
    - 2자 이상: 완전 일치만 허용
    - 4자 이상: 매핑 키에 포함되거나 포함하는 부분 일치도 허용
    yolo_label이 주어지면 후보가 여러 개일 때 우선권을 줍니다.
    """
    if not ocr_norm or len(ocr_norm) < 2:
        return None

    def _resolve(candidates):
        if len(candidates) == 1:
            return candidates[0]
        if yolo_label in candidates:
            return yolo_label
        return None

    # 1) 완전 일치
    if ocr_norm in mapping:
        return _resolve(mapping[ocr_norm])

    # 2) 부분 일치 — OCR 결과가 매핑 키의 부분 문자열인 경우만 허용 (3자 이상)
    #   · ocr_norm in key : OCR이 각인 일부만 읽은 경우
    #                       예) OCR="TYLEN",    key="TYLENOL"    → 허용
    #                       예) OCR="SEROQUEL", key="SEROQUEL100" → 허용
    #   · key in ocr_norm 방향은 허용하지 않음
    #     매핑 코드가 OCR 결과보다 짧으면 노이즈·오탐으로 간주
    #                       예) OCR="ZD4522200", key="ZD452220" → 차단
    if len(ocr_norm) >= 3:
        hits = []
        for key, candidates in mapping.items():
            if ocr_norm in key:
                hits.extend(candidates)
        if hits:
            return _resolve(list(dict.fromkeys(hits)))

    return None


def correct_predictions(
    img,
    result: dict,
    class_names: dict,
    print_mapping: dict,
    conf_thr: float = 0.25,
    img_name: str = "",
) -> dict:
    """
    WBF/단일 모델 예측 결과에 OCR 각인 보정을 적용합니다.

    동작:
      - score < OCR_TRIGGER_CONF(0.6) 인 박스에만 OCR 실행
      - OCR 텍스트가 매핑 테이블의 클래스와 일치하면:
          · 해당 클래스로 확정 (YOLO 클래스와 달라도 교체)
          · score를 OCR 인식 신뢰도로 교체
      - 매핑 실패 시 원본 유지

    Args:
        img          : BGR numpy 이미지
        result       : {"boxes": [[x1n,y1n,x2n,y2n],...], "scores":[...], "labels":[...]}
                       좌표는 0~1 정규화
        class_names  : {class_idx: name}
        print_mapping: build_print_mapping() 결과
        conf_thr     : 참고용 (draw_result 기준과 동일)

    Returns:
        보정된 result dict (원본 불변)
    """
    h, w = img.shape[:2]
    boxes  = list(result["boxes"])
    scores = list(result["scores"])
    labels = list(result["labels"])

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if score >= OCR_TRIGGER_CONF and label not in CONFUSING_CLASSES:
            continue  # 신뢰도 충분 & 헷갈리는 클래스 아님 → OCR 생략

        # ── 크롭 (패딩 포함) ──────────────────────
        x1n, y1n, x2n, y2n = box
        bw = (x2n - x1n) * w
        bh = (y2n - y1n) * h
        x1 = max(0, int(x1n * w - bw * CROP_PAD))
        y1 = max(0, int(y1n * h - bh * CROP_PAD))
        x2 = min(w, int(x2n * w + bw * CROP_PAD))
        y2 = min(h, int(y2n * h + bh * CROP_PAD))

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # ── OCR 실행 ──────────────────────────────
        try:
            ocr_results = read_text_from_crop(crop)  # [(text, ocr_conf), ...]
        except Exception as e:
            print(f"  [OCR 오류] {e}")
            continue

        if not ocr_results:
            continue

        # 개별 토큰 탐색 — OCR 인식 신뢰도 함께 추적
        # 앞뒤 결과를 합치면 "3"+"0"→"30" 같은 오매핑 발생하므로 개별 탐색
        matched_class    = None
        matched_ocr_conf = 0.0
        matched_text     = ""
        for text, ocr_conf in ocr_results:
            candidate = _find_match(_normalize(str(text)), print_mapping, yolo_label=label)
            if candidate is not None:
                matched_class    = candidate
                matched_ocr_conf = float(ocr_conf)
                matched_text     = str(text)
                break

        if matched_class is None:
            continue

        # OCR 신뢰도가 현재 신뢰도보다 0.6 이상 낮으면 보정 생략
        if score - matched_ocr_conf >= 0.6:
            prefix    = f"({img_name}) " if img_name else ""
            orig_name = class_names.get(label, f"cls_{label}")
            print(f"  [OCR 생략] {prefix}{orig_name}  "
                  f"YOLO/Stage2={score:.2f}  OCR={matched_ocr_conf:.2f}  (차이 {score - matched_ocr_conf:.2f} ≥ 0.6)")
            continue

        # ── OCR 매핑 성공 → 클래스·신뢰도를 OCR 결과로 교체 ──
        orig_name = class_names.get(label, f"cls_{label}")
        new_name  = class_names.get(matched_class, f"cls_{matched_class}")
        prefix    = f"({img_name}) " if img_name else ""

        if matched_class != label:
            print(f"  [OCR 보정] {prefix}{orig_name} → {new_name} "
                  f"(각인: {matched_text}) conf {score:.2f} → {matched_ocr_conf:.2f}")
            labels[i] = matched_class
        elif score < OCR_TRIGGER_CONF:
            print(f"  [OCR 확인] {prefix}{orig_name} 일치 "
                  f"(각인: {matched_text}) conf {score:.2f} → {matched_ocr_conf:.2f}")

        scores[i] = matched_ocr_conf

    return {"boxes": boxes, "scores": scores, "labels": labels}
