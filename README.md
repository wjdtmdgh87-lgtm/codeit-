# 알약 객체 검출 및 분류 파이프라인 (2-Stage + OCR)

경구약제 이미지에서 알약의 위치(바운딩 박스)를 검출하고, 복잡한 알약을 정확히 분류하기 위해 **YOLOv12n + EfficientNet-B3 + EasyOCR**을 결합한 하이브리드 객체 검출 프로젝트입니다.

- **데이터셋:** v11 / 73클래스 경구약제
- **YOLO 모델:** `yolo12n.pt` (파라미터 ~2.59M × 5 fold)
- **Stage 2 분류기:** EfficientNet-B3 (파라미터 ~10.9M)

---

## 🚀 주요 기능 및 파이프라인

1. **Stage 1 (YOLOv12n):** 빠르고 정확한 1차 객체 탐지 및 바운딩 박스 추출
2. **Stage 2 (EfficientNet-B3):** YOLO의 예측 신뢰도(Confidence)가 낮은 알약이나 비슷한 모양의 알약들을 잘라내어(Crop) 2차 정밀 분류
3. **OCR 각인 보정 (EasyOCR):** 알약 표면의 텍스트(각인)를 4방향 물리적 회전으로 꼼꼼히 읽어내어, 매핑 DB와 대조를 통해 오탐지 완벽 보정
4. **WBF 앙상블:** 5-Fold 학습을 통해 얻은 5개의 모델 결과를 Weighted Boxes Fusion으로 병합하여 안정성 극대화

---

## 💻 실행 방법

프로젝트 루트(`main.py`가 있는 폴더)에서 실행합니다.

```bash
python main.py --mode [모드] [옵션]
```

---

## 🛠 실행 모드 (`--mode`)

### 1. 데이터 준비 및 1차 학습 (YOLO)
* **`data`**: 데이터팀이 제공한 split txt 파일의 경로를 수정하고 통계를 출력합니다.
* **`train`**: `data.yaml` 기준으로 fold 1 단일 학습을 실행합니다.
* **`train_all`**: fold 1~5를 순서대로 자동 학습합니다. (WBF 앙상블 사용 시 필수)

### 2. Stage 2 (분류기) 학습
* **`crop_data`**: YOLO 1차 학습된 모델(`best_model.pt`)을 이용해 학습용 알약 크롭(Crop) 이미지를 생성합니다.
* **`train_stage2`**: 생성된 크롭 이미지를 이용해 Stage 2 분류기(EfficientNet-B3)를 학습합니다. (결과물: `models/stage2_best.pt`)

### 3. 추론 및 예측
* **`predict`**: 단일 모델로 이미지를 예측합니다. (자동으로 Stage 2 및 OCR 보정이 적용됩니다.)
* **`wbf`**: 5-Fold WBF(Weighted Boxes Fusion) 앙상블 예측을 실행합니다. `models/` 폴더의 `fold*_best.pt` 파일들을 자동으로 사용합니다.

---

## ⚙️ 주요 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--mode` | 실행 모드 | `all` |
| `--source` | 예측할 이미지 경로 또는 폴더 | `None` |
| `--weights` | 예측 시 사용할 가중치 경로 | `models/best_model.pt` |
| `--conf` | 신뢰도 임계값 | `0.25` |
| `--no-ocr` | **OCR 각인 보정 비활성화** | 사용(False) |
| `--no-stage2` | **Stage 2 분류기 비활성화** | 사용(False) |

---

## ⚙️ 현재 학습 설정 (`src/config.py`)

| 항목 | 값 |
|------|----|
| 베이스 모델 | `yolo12n.pt` |
| 이미지 크기 | 832 |
| 배치 크기 | 8 |
| Epochs | 150 (patience=30) |
| 옵티마이저 | SGD (lr=0.01) |
| K-Fold | 5 (seed=42) |
| Stage 2 bypass | conf ≥ 0.85 → Stage 2 생략 |

---

## 💡 실행 예시

**단일 모델 예측 (기본 파이프라인):**
```bash
python main.py --mode predict --source data/images/test/
```

**WBF 앙상블 예측 (OCR 및 Stage 2 끄기):**
```bash
python main.py --mode wbf --source data/images/test/ --no-ocr --no-stage2
```

---

## 📁 결과 저장 위치

```text
models/
├── baseline_{모델명}_best.pt    # train 모드 결과
├── fold1_{모델명}_best.pt       # train_all 모드 (Fold 1~5)
├── ...
└── stage2_best.pt              # train_stage2 모드 결과

runs/detect/
├── predict_{모델명}/            # predict 모드 예측 결과 이미지
└── wbf_{모델명}/                # wbf 모드 예측 결과 이미지

results/
├── stage2/                     # Stage 2 학습 로그 및 가중치
└── {실험명}/                    # YOLO 학습 로그
```

---

## 📦 설치 및 요구사항

**macOS / CPU-only:**
```bash
pip install -r requirements.txt
```

**Windows / Linux + NVIDIA GPU:**
```bash
pip install -r requirements.txt
pip install -r requirements-cuda.txt   # torch+cu128 오버라이드
```

> `ensemble-boxes`, `easyocr` 는 `requirements.txt` 에 포함되어 있어 별도 설치 불필요합니다.

### 주요 의존성

| 패키지 | 버전 |
|--------|------|
| `ultralytics` | 8.4.40 |
| `torch` | 2.11.0 (+cu128 for GPU) |
| `easyocr` | 1.7.2 |
| `ensemble-boxes` | 1.0.9 |
| `opencv-python` | 4.13.0 |
