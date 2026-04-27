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

from config import ROOT, DATA_DIR, ANNOT_DIR


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

    # 이미 train/ 포함 → 그대로
    if "train" in parts:
        return line.strip()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="특정 txt 파일만 처리")
    args = parser.parse_args()

    if args.input:
        process_file(Path(args.input))
    else:
        build()