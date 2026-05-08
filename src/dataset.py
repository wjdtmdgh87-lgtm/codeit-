"""
src/dataset.py
txt 어노테이션 기준 Stratified K-Fold 분할

역할:
  - train_images/ + train_labels/*.txt 를 읽어
  - Stratified K-Fold 로 train/val 경로 목록 txt 생성
    → data/splits/fold{N}_train.txt
    → data/splits/fold{N}_val.txt
  - dataset.yaml 의 train/val 경로가 이 txt 를 가리킴
"""

import random
from pathlib import Path
from collections import defaultdict

from config import TRAIN_IMG_DIR, ANNOT_DIR, KFOLD, ROOT


SPLITS_DIR = ROOT / "data" / "splits"


def build():
    """split txt 파일 생성"""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # train_labels/ 의 txt 파일 목록 수집
    label_map = {}
    for txt in sorted(ANNOT_DIR.glob("*.txt")):
        cls_set = {int(l.split()[0]) for l in txt.read_text().splitlines() if l.strip()}
        label_map[txt.stem] = cls_set

    print(f"  어노테이션 {len(label_map)}개 발견")

    # 이미지 파일 존재 여부 확인
    valid = {s for s in label_map if _find_img(s) is not None}
    print(f"  이미지 매칭 {len(valid)}장")

    if not valid:
        print("  [오류] 매칭되는 이미지가 없습니다.")
        print(f"  TRAIN_IMG_DIR = {TRAIN_IMG_DIR}")
        print(f"  ANNOT_DIR     = {ANNOT_DIR}")
        return

    # Stratified K-Fold
    cfg    = KFOLD
    random.seed(cfg["seed"])

    groups = defaultdict(list)
    for stem in valid:
        key = tuple(sorted(label_map[stem]))
        groups[key].append(stem)

    folds = [[] for _ in range(cfg["n_folds"])]
    for stems in groups.values():
        random.shuffle(stems)
        for i, s in enumerate(stems):
            folds[i % cfg["n_folds"]].append(s)

    # 모든 fold 의 txt 저장
    for fold_idx in range(cfg["n_folds"]):
        val_set = set(folds[fold_idx])
        trn_set = {s for i, f in enumerate(folds) if i != fold_idx for s in f}

        _write_split_txt(fold_idx + 1, "train", trn_set)
        _write_split_txt(fold_idx + 1, "val",   val_set)

    use = cfg["use_fold"]
    val_n = len(folds[use])
    trn_n = len(valid) - val_n
    print(f"  Fold {use + 1} 기준 - Train {trn_n}장 / Val {val_n}장")
    print(f"  저장 위치: {SPLITS_DIR}")


def _write_split_txt(fold_n: int, split: str, stems: set):
    """이미지 절대경로 목록을 txt 로 저장"""
    lines = []
    for s in sorted(stems):
        img = _find_img(s)
        if img:
            lines.append(str(img.resolve()))

    path = SPLITS_DIR / f"fold{fold_n}_{split}.txt"
    path.write_text("\n".join(lines), encoding="utf-8")


def _find_img(stem: str):
    for ext in (".png", ".jpg", ".jpeg"):
        p = TRAIN_IMG_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


if __name__ == "__main__":
    print("=== 데이터 분할 ===")
    build()