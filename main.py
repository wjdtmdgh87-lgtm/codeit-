"""
main.py  ← project_root/ 에 위치해야 합니다.

사용법:
  python main.py --mode data       # 데이터 경로 수정 + 통계 출력
  python main.py --mode train      # fold 1 단일 학습
  python main.py --mode train_all  # fold 1~5 전체 자동 학습
  python main.py --mode predict    # 단일 모델 예측  (--source 필수)
  python main.py --mode wbf        # 5-Fold WBF 앙상블 예측  (--source 필수)
  python main.py --mode all        # data + train 순서대로
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=["data", "train", "train_all", "predict", "wbf", "all"], default="all")
    parser.add_argument("--source",  default=None, help="predict/wbf 모드 시 이미지 경로")
    parser.add_argument("--weights", default=None, help="predict 모드 시 가중치 경로")
    parser.add_argument("--conf",    type=float, default=0.25)
    args = parser.parse_args()

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

    if args.mode == "predict":
        if not args.source:
            print("[오류] --source 경로를 지정해주세요.")
            return
        from model import predict
        print("\n=== 예측 ===")
        predict(args.source, weights=args.weights, conf=args.conf)

    if args.mode == "wbf":
        if not args.source:
            print("[오류] --source 경로를 지정해주세요.")
            return
        from wbf_ensemble import wbf_predict
        print("\n=== WBF 앙상블 예측 ===")
        wbf_predict(args.source, conf=args.conf)


if __name__ == "__main__":
    main()
