"""
Split 구조 검증 스크립트 (새 위치: data/v3)
- splits/ 와 oversampled/ 파일 존재 여부
- 각 폴드별 구성 (normal/rare/emg/new/crop)
- missing 파일 확인
"""
import sys
from pathlib import Path
import collections

# 프로젝트 루트 기준으로 자동 경로 설정
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import ROOT

BASE = ROOT / "data"
IMG_DIR = BASE / 'images' / 'train'
LBL_DIR = BASE / 'labels'
SPLIT_DIR = BASE / 'splits'
OVER_DIR = SPLIT_DIR / 'oversampled'

# class 분류
EMERGENCY_CLS = {6, 10, 15, 17, 19, 20, 21, 24, 47, 48}
NEW_CLS = set(range(56, 67))  # 56~66

# code_mapping에서 bbox 수 읽기 (희귀클래스 판별용)
import csv
rare_cls = set()
try:
    with open(BASE / 'code_mapping.csv', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ci = int(row.get('class_index', row.get('cls', -1)))
                bbox = int(row.get('bbox_count', row.get('bbox', 0)))
                if 0 <= ci <= 55 and bbox <= 10:
                    rare_cls.add(ci)
            except:
                pass
except Exception as e:
    print(f"[WARN] code_mapping.csv 읽기 실패: {e}")

print(f"BASE: {BASE}")
print(f"Emergency classes: {sorted(EMERGENCY_CLS)}")
print(f"New classes (56-66): {sorted(NEW_CLS)}")
print(f"Rare classes (bbox<=10, cls 0-55): {sorted(rare_cls)}")
print()

def classify_image(img_path_str):
    p = Path(img_path_str)
    stem = p.stem
    # crop 이미지: _cls{n}_b{n} 패턴
    if '_cls' in stem and '_b' in stem:
        return 'crop'
    # 라벨 파일에서 클래스 확인
    lbl = LBL_DIR / (stem + '.txt')
    if not lbl.exists():
        return 'unknown'
    classes_in_label = set()
    try:
        for line in lbl.read_text(encoding='utf-8').splitlines():
            parts = line.strip().split()
            if parts:
                classes_in_label.add(int(parts[0]))
    except:
        return 'unknown'
    if classes_in_label & NEW_CLS:
        return 'new'
    if classes_in_label & EMERGENCY_CLS:
        return 'emergency'
    if classes_in_label & rare_cls:
        return 'rare'
    return 'normal'

print("=" * 70)
print("[ SPLIT 파일 검증 ]")
print("=" * 70)

for fold in range(1, 6):
    for split_type in ['train', 'val']:
        fpath = SPLIT_DIR / f'fold{fold}_{split_type}.txt'
        if not fpath.exists():
            print(f"[MISSING] {fpath.name}")
            continue
        lines = [l.strip() for l in fpath.read_text(encoding='utf-8').splitlines() if l.strip()]
        missing = []
        for l in lines:
            # 경로가 ./images/... 형식이면 BASE 기준으로 해석
            p = BASE / l.lstrip('./')
            if not p.exists():
                missing.append(l)
        print(f"fold{fold}_{split_type}: {len(lines)}장, missing={len(missing)}")
        if missing[:3]:
            for m in missing[:3]:
                print(f"  예시 missing: {m}")

print()
print("=" * 70)
print("[ OVERSAMPLED 파일 검증 ]")
print("=" * 70)

total_ok = True
for fold in range(1, 6):
    fpath = OVER_DIR / f'fold{fold}_train_oversampled.txt'
    if not fpath.exists():
        print(f"[MISSING] {fpath.name}")
        total_ok = False
        continue
    lines = [l.strip() for l in fpath.read_text(encoding='utf-8').splitlines() if l.strip()]
    missing = []
    counter = collections.Counter()
    for l in lines:
        p = BASE / l.lstrip('./')
        if not p.exists():
            missing.append(l)
        cat = classify_image(l)
        counter[cat] += 1
    print(f"fold{fold}_train_oversampled: {len(lines)}줄, missing={len(missing)}")
    print(f"  구성: normal={counter['normal']}, rare={counter['rare']}, "
          f"emergency={counter['emergency']}, new={counter['new']}, "
          f"crop={counter['crop']}, unknown={counter['unknown']}")
    if missing[:3]:
        total_ok = False
        for m in missing[:3]:
            print(f"  예시 missing: {m}")

print()
if total_ok:
    print("==> 모든 oversampled 파일 OK (missing=0)")
else:
    print("==> 일부 파일 누락 있음. 위 내용 확인 필요.")
