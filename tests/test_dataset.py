"""
테스트 구글 코랩(Colab)
tests/test_dataset.py
txt 어노테이션 기준 Stratified K-Fold 분할 + Oversampling (오버샘플링)
"""

import random
from pathlib import Path
from collections import defaultdict, Counter

# test_config.py에서 환경설정을 가져옵니다.
from test_config import TRAIN_IMG_DIR, ANNOT_DIR, KFOLD, DATA_DIR

# 🌟 수정 포인트 1: 오버샘플링된 파일들을 모아둘 전용 폴더를 만듭니다.
OVERSAMPLED_DIR = DATA_DIR / "splits" / "oversampled"

def build():
    """split txt 파일 생성"""
    OVERSAMPLED_DIR.mkdir(parents=True, exist_ok=True)

    label_map = {}
    for txt in sorted(ANNOT_DIR.glob("*.txt")):
        cls_set = {int(l.split()[0]) for l in txt.read_text().splitlines() if l.strip()}
        label_map[txt.stem] = cls_set

    print(f"  어노테이션 {len(label_map)}개 발견")

    valid = {s for s in label_map if _find_img(s) is not None}
    print(f"  이미지 매칭 {len(valid)}장")

    if not valid:
        print("  [오류] 매칭되는 이미지가 없습니다.")
        return

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

    for fold_idx in range(cfg["n_folds"]):
        val_set = list(folds[fold_idx])
        trn_set = [s for i, f in enumerate(folds) if i != fold_idx for s in f]

        # 🌟 수정 포인트 2: 학습용 데이터(trn_set)에 오버샘플링 마법을 적용합니다!
        # (주의: 테스트용인 val_set은 절대 뻥튀기하면 안 됩니다. 정직하게 평가해야 하니까요!)
        trn_oversampled = apply_oversampling(trn_set, label_map)

        # 오버샘플링된 학습용 파일 저장 (이름 뒤에 _oversampled가 붙습니다)
        _write_split_txt(fold_n=fold_idx + 1, split="train_oversampled", stems=trn_oversampled)
        # 원본 유지된 테스트용 파일 저장
        _write_split_txt(fold_n=fold_idx + 1, split="val", stems=val_set)

    use = cfg["use_fold"]
    val_n = len(folds[use])
    trn_n_original = len(valid) - val_n

    # 우리가 만든 오버샘플링 파일의 실제 줄(Line) 수를 읽어옵니다.
    sample_txt = OVERSAMPLED_DIR / f"fold{use + 1}_train_oversampled.txt"
    trn_n_oversampled = len(sample_txt.read_text().strip().split('\n'))

    print(f"  Fold {use + 1} 기준")
    print(f"  - 원본 Train 데이터 : {trn_n_original}장")
    print(f"  - 🚀 오버샘플링 후 Train 데이터 : {trn_n_oversampled}장 (뻥튀기 완료!)")
    print(f"  - Val 데이터 (원본 유지) : {val_n}장")
    print(f"  저장 위치: {OVERSAMPLED_DIR}")


# =====================================================================
# 🌟 신규 함수: 희귀 알약 사진을 복사해서 늘려주는 오버샘플링 로직!
# =====================================================================
def apply_oversampling(trn_set, label_map):
    # 1. 현재 훈련 세트 안에 있는 모든 알약의 개수를 셉니다.
    class_counts = Counter()
    for stem in trn_set:
        class_counts.update(label_map[stem])

    # 2. 가장 많이 등장한 '흔한 알약'의 개수를 찾습니다. (이게 우리의 목표 개수입니다)
    max_count = max(class_counts.values()) if class_counts else 0

    oversampled_list = []
    for stem in trn_set:
        # 이 사진 안에 있는 알약들 중, 가장 '희귀한' 알약이 몇 번 등장했는지 찾습니다.
        rarest_count = min([class_counts[c] for c in label_map[stem]])

        # 목표 개수를 맞추기 위해 이 사진을 몇 번 복사할지 계산합니다.
        repeat_times = int(max_count / rarest_count)

        # 무한정 복사되는 것을 막기 위한 안전장치! (최대 10배까지만 뻥튀기)
        repeat_times = min(repeat_times, 10)
        repeat_times = max(1, repeat_times)  # 무조건 최소 1번은 들어감

        # 계산된 횟수만큼 리스트에 중복해서 사진 이름을 넣습니다.
        oversampled_list.extend([stem] * repeat_times)

    # 복사본들이 골고루 섞이도록 한 번 섞어줍니다.
    random.shuffle(oversampled_list)
    return oversampled_list


def _write_split_txt(fold_n: int, split: str, stems: list):
    lines = []
    # 중복을 허용해야 하므로 기존의 sorted(set(stems)) 대신 그냥 리스트를 순회합니다.
    for s in stems:
        img = _find_img(s)
        if img:
            lines.append(str(img.resolve()))

    path = OVERSAMPLED_DIR / f"fold{fold_n}_{split}.txt"
    path.write_text("\n".join(lines), encoding="utf-8")


def _find_img(stem: str):
    for ext in (".png", ".jpg", ".jpeg"):
        p = TRAIN_IMG_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


if __name__ == "__main__":
    print("=== 데이터 분할 및 오버샘플링 시작 ===")
    build()