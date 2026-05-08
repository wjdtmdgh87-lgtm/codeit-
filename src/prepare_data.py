"""
[추가 사항]
- sprint_ai_project1_data/train_annotations 아래의 JSON 파일들을 읽어서
  YOLO 형식 txt 라벨 파일로 변환합니다.
- data/labels/train 에 txt 라벨을 생성합니다.
- data.yaml 파일도 함께 생성합니다.
- 이미지는 data/images/train, data/images/test 구조를 사용한다고 가정합니다.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "sprint_ai_project1_data"
RAW_ANN_DIR = RAW_DIR / "train_annotations"

DATA_DIR = ROOT / "data"
TRAIN_IMG_DIR = DATA_DIR / "images" / "train"
TEST_IMG_DIR = DATA_DIR / "images" / "test"
LABEL_DIR = DATA_DIR / "labels" / "train"

DATASET_YAML = ROOT / "data.yaml"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def xywh_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return x_center, y_center, w_norm, h_norm


def build_category_mapping():
    """
    전체 JSON을 훑어서 category_id -> class_index 매핑 생성
    YOLO는 class id가 0부터 연속 정수여야 하므로 재매핑합니다.
    """
    category_map = {}

    for json_path in RAW_ANN_DIR.rglob("*.json"):
        try:
            data = load_json(json_path)
        except Exception:
            continue

        categories = data.get("categories", [])
        for cat in categories:
            cat_id = cat.get("id")
            cat_name = cat.get("name", str(cat_id))
            if cat_id is not None and cat_id not in category_map:
                category_map[cat_id] = cat_name

    sorted_cat_ids = sorted(category_map.keys())
    cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(sorted_cat_ids)}
    class_names = [category_map[cat_id] for cat_id in sorted_cat_ids]

    return cat_id_to_idx, class_names


def build_labels():
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    cat_id_to_idx, class_names = build_category_mapping()

    json_count = 0
    label_count = 0
    matched_image_count = 0

    for json_path in RAW_ANN_DIR.rglob("*.json"):
        try:
            data = load_json(json_path)
        except Exception:
            continue

        json_count += 1

        images = data.get("images", [])
        annotations = data.get("annotations", [])

        if not images or not annotations:
            continue

        image_info = images[0]
        ann = annotations[0]

        file_name = image_info.get("file_name")
        img_w = image_info.get("width")
        img_h = image_info.get("height")
        bbox = ann.get("bbox")
        category_id = ann.get("category_id")

        if not file_name or img_w is None or img_h is None:
            continue
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        if category_id not in cat_id_to_idx:
            continue

        image_path = TRAIN_IMG_DIR / file_name
        if not image_path.exists():
            # 이미지가 없으면 라벨도 만들지 않음
            continue

        matched_image_count += 1

        class_idx = cat_id_to_idx[category_id]
        x_center, y_center, w_norm, h_norm = xywh_to_yolo(bbox, img_w, img_h)

        txt_name = Path(file_name).stem + ".txt"
        txt_path = LABEL_DIR / txt_name

        line = f"{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
        txt_path.write_text(line, encoding="utf-8")
        label_count += 1

    return class_names, {
        "json_count": json_count,
        "label_count": label_count,
        "matched_image_count": matched_image_count,
    }


def build_data_yaml(class_names):
    text = [
        f"path: {ROOT}",
        "train: data/images/train",
        "val: data/images/train",
        "test: data/images/test",
        "names:",
    ]

    for idx, name in enumerate(class_names):
        safe_name = str(name).replace('"', "'")
        text.append(f"  {idx}: \"{safe_name}\"")

    DATASET_YAML.write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    print("=== YOLO 라벨 생성 시작 ===")
    print(f"RAW_ANN_DIR   = {RAW_ANN_DIR}")
    print(f"TRAIN_IMG_DIR = {TRAIN_IMG_DIR}")
    print(f"LABEL_DIR     = {LABEL_DIR}")

    class_names, stats = build_labels()
    build_data_yaml(class_names)

    print("\n=== 생성 완료 ===")
    print(f"클래스 수           : {len(class_names)}")
    print(f"읽은 JSON 수        : {stats['json_count']}")
    print(f"매칭된 이미지 수    : {stats['matched_image_count']}")
    print(f"생성된 txt 라벨 수  : {stats['label_count']}")
    print(f"data.yaml 생성 위치 : {DATASET_YAML}")


if __name__ == "__main__":
    main()