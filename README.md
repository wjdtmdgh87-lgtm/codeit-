# 알약 객체 검출 파이프라인

경구약제 이미지에서 알약의 위치(바운딩 박스)와 클래스(약품명)를 검출하는 YOLOv11 기반 객체 검출 프로젝트입니다.

---

## 실행 방법

프로젝트 루트(`main.py`가 있는 폴더)에서 실행합니다.

```bash
python main.py --mode [모드] [옵션]
```

---

## 실행 모드

### `--mode data`
데이터팀이 제공한 split txt 파일의 경로를 수정하고 통계를 출력합니다.

```bash
python main.py --mode data
```

### `--mode train`
`data.yaml` 기준으로 fold 1 단일 학습을 실행합니다.  
학습 완료 후 `models/baseline_{모델명}_best.pt`에 저장됩니다.

```bash
python main.py --mode train
```

### `--mode train_all`
fold 1~5를 순서대로 자동 학습합니다.  
각 fold 결과는 `models/fold{N}_{모델명}_best.pt`에 저장됩니다.  
WBF 앙상블 사용 시 이 모드로 먼저 학습해야 합니다.

```bash
python main.py --mode train_all
```

### `--mode predict`
단일 모델로 이미지를 예측합니다.  
결과는 `runs/detect/predict_{모델명}/`에 저장됩니다.

```bash
python main.py --mode predict --source data/images/test/

# 가중치 직접 지정
python main.py --mode predict --source data/images/test/ --weights models/best_model.pt

# 신뢰도 임계값 조정
python main.py --mode predict --source data/images/test/ --conf 0.5
```

### `--mode wbf`
5-Fold WBF(Weighted Boxes Fusion) 앙상블 예측을 실행합니다.  
`models/` 폴더의 `fold*_best.pt` 파일을 자동으로 수집해서 예측합니다.  
결과는 `runs/detect/wbf_{모델명}/`에 저장됩니다.

```bash
python main.py --mode wbf --source data/images/test/

# 신뢰도 임계값 조정
python main.py --mode wbf --source data/images/test/ --conf 0.15
```

### `--mode all`
데이터 준비(`data`) → 학습(`train`)을 순서대로 실행합니다.

```bash
python main.py --mode all
```

---

## 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--mode` | 실행 모드 | `all` |
| `--source` | 예측 이미지 경로 또는 폴더 | `None` |
| `--weights` | 예측 시 사용할 가중치 경로 | `models/best_model.pt` |
| `--conf` | 신뢰도 임계값 | `0.25` |

---

## 결과 저장 위치

```
models/
├── baseline_{모델명}_best.pt    # train 모드 결과
├── fold1_{모델명}_best.pt       # train_all 모드 결과
├── fold2_{모델명}_best.pt
└── ...

runs/detect/
├── predict_{모델명}/            # predict 모드 결과
└── wbf_{모델명}/                # wbf 모드 결과

results/
└── {실험명}/
    ├── weights/best.pt
    ├── results.csv
    └── confusion_matrix.png
```

---

## 설치

```bash
pip install -r requirements.txt
```

WBF 앙상블 사용 시 추가 설치가 필요합니다.

```bash
pip install ensemble-boxes
```