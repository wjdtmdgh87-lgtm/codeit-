"""
main.py  ← project_root/ 에 위치해야 합니다.

사용법:
  python main.py --mode data     # 데이터 변환 + 분할
  python main.py --mode train    # 학습
  python main.py --mode predict  # 예측  (--source 필수)
  python main.py --mode all      # data + train 순서대로
"""

import sys
import argparse
from pathlib import Path

# src/ 를 모듈 경로에 추가 (main.py 는 project_root/ 에서 실행)
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=["data", "train", "predict", "all"], default="all")
    parser.add_argument("--source",  default=None, help="predict 모드 시 이미지 경로")
    parser.add_argument("--weights", default=None, help="predict 모드 시 가중치 경로")
    parser.add_argument("--conf",    type=float, default=0.25)
    args = parser.parse_args()

    if args.mode in ("data", "all"):
        from dataset import build
        print("=== 데이터 준비 ===")
        build()

    if args.mode in ("train", "all"):
        from model import train
        print("\n=== 학습 시작 ===")
        train()

    if args.mode == "predict":
        if not args.source:
            print("[오류] --source 경로를 지정해주세요.")
            print("  예) python main.py --mode predict --source data/images/test/")
            return
        from model import predict
        print("\n=== 예측 ===")
        predict(args.source, weights=args.weights, conf=args.conf)


if __name__ == "__main__":
    main()