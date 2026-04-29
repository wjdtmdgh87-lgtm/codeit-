"""
src/dataset.py
데이터팀이 제공한 split txt 파일을 처리합니다.

역할:
  - 데이터팀 txt 파일의 경로에 train/ 이 없으면 자동 추가
  - 각 fold별 데이터 통계 출력
  - data/splits/ 에 최종 txt 저장

사용법:
  python dataset.py --input data/splits/fold1_train.txt
  python dataset.py  # splits/ 폴더 전체 처리
"""

import argparse
from pathlib import Path
from collections import defaultdict, Counter

from config import ROOT, DATA_DIR, ANNOT_DIR, CROPS_DIR, RARE_CLASS_THR, CROPS_PER_CLASS


SPLITS_DIR  = DATA_DIR / "splits"
IMG_DIR     = DATA_DIR / "images" / "train"


def fix_path(line: str) -> str:
    """
    데이터팀 txt 경로에 train/ 이 없으면 추가합니다.
    data/images/K-...png → data/images/train/K-...png
    ./images/K-...png    → data/images/train/K-...png
    data/images/train/K-...png → 그대로 유지
    """
    p = Path(line.strip())
    parts = p.parts

    # 이미 train/ 포함 → 슬래시 정규화 후 반환 (Windows 백슬래시 대응)
    if "train" in parts:
        return line.strip().replace("\\", "/")

    # ./images/K-... 또는 data/images/K-... 형태
    fname = p.name
    return f"data/images/train/{fname}"


def get_stats(lines: list, label_dir: Path) -> dict:
    """txt 파일의 통계를 반환합니다."""
    unique   = set()
    cls_counter = Counter()

    for line in lines:
        fname = Path(line.strip()).name
        stem  = Path(fname).stem
        unique.add(stem)

        lbl = label_dir / f"{stem}.txt"
        if lbl.exists():
            for l in lbl.read_text().splitlines():
                parts = l.strip().split()
                if parts:
                    cls_counter[int(parts[0])] += 1

    return {
        "total_lines":   len(lines),
        "unique_images": len(unique),
        "classes":       len(cls_counter),
        "cls_counter":   cls_counter,
    }


def process_file(input_path: Path, output_path: Path = None):
    """단일 txt 파일 처리 — 경로 수정 + 통계 출력"""
    lines = [l for l in input_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    # 경로 수정
    fixed = [fix_path(l) for l in lines]

    # 저장
    out = output_path or input_path
    out.write_text("\n".join(fixed), encoding="utf-8")

    # 통계
    stats = get_stats(fixed, ANNOT_DIR)
    print(f"\n  {input_path.name}")
    print(f"    총 줄 수      : {stats['total_lines']}줄")
    print(f"    고유 이미지   : {stats['unique_images']}장")
    print(f"    등장 클래스   : {stats['classes']}종")

    # 누락 이미지 확인
    missing = []
    for line in fixed:
        img = ROOT / line
        if not img.exists():
            missing.append(line)
    if missing:
        print(f"    누락 이미지   : {len(missing)}개")
        for m in missing[:3]:
            print(f"      예시: {m}")
    else:
        print(f"    누락 이미지   : 없음 ✓")


def build():
    """splits/ 폴더의 모든 txt 파일 처리"""
    txt_files = sorted(SPLITS_DIR.glob("*.txt"))
    if not txt_files:
        print(f"[오류] {SPLITS_DIR} 에 txt 파일이 없습니다.")
        return

    print(f"=== 데이터팀 split txt 처리 ({len(txt_files)}개) ===")
    for f in txt_files:
        process_file(f)
    print(f"\n완료. 저장 위치: {SPLITS_DIR}")


def generate_crops(val_ratio: float = 0.2, crop_pad: float = 0.15, seed: int = 42):
    """
    YOLO 어노테이션에서 pill crop을 추출해 data/crops/{train,val}/class_idx/ 에 저장합니다.

    - 클래스별 stratified split으로 val 균등 배분
    - train 크롭이 CROPS_PER_CLASS 미만이면 증강으로 패딩 (클래스 균형)
    - rare class (≤RARE_CLASS_THR)는 기본 증강도 적용

    val_ratio : 각 클래스별 val 이미지 비율 (기본 20%)
    crop_pad  : bbox 패딩 비율
    seed      : 재현성을 위한 random seed
    """
    import cv2
    import random
    random.seed(seed)

    all_imgs = sorted(IMG_DIR.glob("*.png")) + sorted(IMG_DIR.glob("*.jpg"))
    if not all_imgs:
        print(f"[크롭] [오류] 이미지 없음: {IMG_DIR}")
        return

    # 클래스별 등장 이미지 목록 수집
    cls_to_imgs: defaultdict = defaultdict(list)
    cls_count: Counter = Counter()
    for img_path in all_imgs:
        lbl = ANNOT_DIR / f"{img_path.stem}.txt"
        if not lbl.exists():
            continue
        seen = set()
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            c = int(parts[0])
            cls_count[c] += 1
            if c not in seen:
                cls_to_imgs[c].append(img_path)
                seen.add(c)

    # 클래스별 stratified split — 각 클래스 이미지의 val_ratio를 val로 배정
    val_stems: set = set()
    for c, imgs in cls_to_imgs.items():
        n_val = max(1, round(len(imgs) * val_ratio))
        val_imgs = random.sample(imgs, n_val)
        val_stems.update(img.stem for img in val_imgs)

    rare_classes = {c for c, cnt in cls_count.items() if cnt <= RARE_CLASS_THR}
    print(f"[크롭] rare class {len(rare_classes)}개 (회전+flip 증강 적용): {sorted(rare_classes)}")

    jpg_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    train_saved: Counter = Counter()  # train 크롭 수 (패딩 전)
    val_saved:   Counter = Counter()
    crops_train = CROPS_DIR / "train"
    crops_val   = CROPS_DIR / "val"

    # ── 1st pass: 실제 크롭 저장 + train 크롭 배열을 클래스별로 수집 ──
    train_pool: defaultdict = defaultdict(list)  # cls_idx → list of np.ndarray

    for img_path in all_imgs:
        lbl = ANNOT_DIR / f"{img_path.stem}.txt"
        if not lbl.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        is_val = img_path.stem in val_stems
        out_base = crops_val if is_val else crops_train

        for i, line in enumerate(lbl.read_text().splitlines()):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_idx = int(parts[0])
            cx, cy, bw, bh = (float(parts[1]), float(parts[2]),
                               float(parts[3]), float(parts[4]))

            x1 = max(0, int((cx - bw / 2 - bw * crop_pad) * w))
            y1 = max(0, int((cy - bh / 2 - bh * crop_pad) * h))
            x2 = min(w, int((cx + bw / 2 + bw * crop_pad) * w))
            y2 = min(h, int((cy + bh / 2 + bh * crop_pad) * h))

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            out_dir = out_base / str(cls_idx)
            out_dir.mkdir(parents=True, exist_ok=True)

            stem = f"{img_path.stem}_crop{i}"
            cv2.imwrite(str(out_dir / f"{stem}.jpg"), crop, jpg_params)

            if is_val:
                val_saved[cls_idx] += 1
            else:
                train_saved[cls_idx] += 1
                train_pool[cls_idx].append(crop)

                # rare class 기본 증강 (train 전용)
                if cls_idx in rare_classes:
                    augments = [
                        ("_r90",  cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)),
                        ("_r180", cv2.rotate(crop, cv2.ROTATE_180)),
                        ("_r270", cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)),
                        ("_fh",   cv2.flip(crop, 1)),
                        ("_fv",   cv2.flip(crop, 0)),
                    ]
                    for suffix, aug in augments:
                        cv2.imwrite(str(out_dir / f"{stem}{suffix}.jpg"), aug, jpg_params)
                        train_saved[cls_idx] += 1
                        train_pool[cls_idx].append(aug)

    # ── 2nd pass: CROPS_PER_CLASS 미달 클래스 → 증강 패딩 ──
    _aug_fns = [
        lambda c: cv2.rotate(c, cv2.ROTATE_90_CLOCKWISE),
        lambda c: cv2.rotate(c, cv2.ROTATE_180),
        lambda c: cv2.rotate(c, cv2.ROTATE_90_COUNTERCLOCKWISE),
        lambda c: cv2.flip(c, 1),
        lambda c: cv2.flip(c, 0),
        lambda c: cv2.rotate(cv2.flip(c, 1), cv2.ROTATE_90_CLOCKWISE),
        lambda c: cv2.rotate(cv2.flip(c, 0), cv2.ROTATE_90_CLOCKWISE),
    ]

    padded_total = 0
    padded_classes = 0
    for cls_idx, pool in train_pool.items():
        need = CROPS_PER_CLASS - train_saved[cls_idx]
        if need <= 0:
            continue

        out_dir = crops_train / str(cls_idx)
        out_dir.mkdir(parents=True, exist_ok=True)

        for k in range(need):
            src = pool[k % len(pool)]
            aug_fn = _aug_fns[k % len(_aug_fns)]
            aug = aug_fn(src)
            cv2.imwrite(str(out_dir / f"pad_{cls_idx}_{k:05d}.jpg"), aug, jpg_params)
        train_saved[cls_idx] += need
        padded_total += need
        padded_classes += 1

    total_train = sum(train_saved.values())
    total_val   = sum(val_saved.values())
    print(f"[크롭] 생성 완료: train {total_train}개 / val {total_val}개, {len(train_saved)}개 클래스")
    print(f"  패딩 추가: {padded_total}개 ({padded_classes}개 클래스, 목표 {CROPS_PER_CLASS}개/클래스)")
    print(f"  저장 위치: {CROPS_DIR}")
    return CROPS_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="특정 txt 파일만 처리")
    args = parser.parse_args()

    if args.input:
        process_file(Path(args.input))
    else:
        build()