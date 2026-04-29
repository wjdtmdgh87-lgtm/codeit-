"""
main.py  ← project_root/ 에 위치해야 합니다.

사용법:
  python main.py --mode data          # 데이터 경로 수정 + 통계 출력
  python main.py --mode train         # fold 1 단일 학습
  python main.py --mode train_all     # fold 1~5 전체 자동 학습
  python main.py --mode crop_data     # Stage 2용 crop 데이터셋 생성
  python main.py --mode train_stage2  # Stage 2 EfficientNet-B3 학습
  python main.py --mode predict --source data/images/test  # 단일 모델 예측
  python main.py --mode wbf --source data/images/test      # 5-Fold WBF 앙상블 예측
  python main.py --mode all           # data + train 순서대로

옵션:
  --no-ocr      OCR 각인 보정 비활성화
  --no-stage2   Stage 2 crop 분류기 비활성화
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["data", "train", "train_all", "crop_data", "train_stage2",
                 "predict", "wbf", "all"],
        default="all",
    )
    parser.add_argument("--source",     default=None,  help="predict/wbf 모드 시 이미지 경로")
    parser.add_argument("--weights",    default=None,  help="predict 모드 시 가중치 경로")
    parser.add_argument("--conf",       type=float, default=0.25)
    parser.add_argument("--no-ocr",     action="store_true", help="OCR 각인 보정 비활성화")
    parser.add_argument("--no-stage2",  action="store_true", help="Stage 2 crop 분류기 비활성화")
    args = parser.parse_args()

    use_ocr    = not args.no_ocr
    use_stage2 = not args.no_stage2

    if args.mode in ("data", "all"):
        from dataset import build
        print("=== 데이터 준비 ===")
        build()

    if args.mode in ("train", "all"):
        from model import train
        print("\n=== 학습 시작 (fold 1) ===")
        train()

    if args.mode == "train_all":
        from model import train_all_folds
        print("\n=== 5-Fold 전체 학습 시작 ===")
        train_all_folds(n_folds=5)

    if args.mode == "crop_data":
        from dataset import generate_crops
        print("\n=== Stage 2 크롭 데이터셋 생성 ===")
        generate_crops()

    if args.mode == "train_stage2":
        from crop_classifier import train_stage2
        print("\n=== Stage 2 학습 ===")
        train_stage2()

    if args.mode == "predict":
        if not args.source:
            print("[오류] --source 경로를 지정해주세요.")
            return
        from model import predict
        print("\n=== 예측 ===")
        predict(args.source, weights=args.weights, conf=args.conf,
                use_ocr=use_ocr, use_stage2=use_stage2)

    if args.mode == "wbf":
        if not args.source:
            print("[오류] --source 경로를 지정해주세요.")
            return
        from wbf_ensemble import wbf_predict
        print("\n=== WBF 앙상블 예측 ===")
        wbf_predict(args.source, conf=args.conf,
                    use_ocr=use_ocr, use_stage2=use_stage2)


if __name__ == "__main__":
    main()
