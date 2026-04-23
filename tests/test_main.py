"""
테스트 구글 코랩(Colab)
tests/test_main.py

사용법:
  python main.py --mode data     # 데이터 변환 + 분할
  python main.py --mode train    # 학습
  python main.py --mode predict  # 예측  (--source 필수)
  python main.py --mode all      # data + train 순서대로
"""

import sys
import argparse
from pathlib import Path
import models.test2_model as t2
import test_dataset as td


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=["data", "train", "predict", "all"], default="all")
    parser.add_argument("--source",  default=None, help="predict 모드 시 이미지 경로")
    parser.add_argument("--weights", default=None, help="predict 모드 시 가중치 경로")
    parser.add_argument("--conf",    type=float, default=0.25)

    # 🌟 수정 1: 모드를 'all'로 설정하고, 예측에 필요한 사진 폴더 경로도 함께 쥐여줍니다!
    args = parser.parse_args(args=['--mode', 'all', '--source', '/content/data/v3/test_images'])

    if args.mode in ("data", "all"):
        # from dataset import build
        print("=== 데이터 준비 ===")
        td.build()

    if args.mode in ("train", "all"):
        # from model import train
        print("\n=== 학습 시작 ===")
        t2.train()

    # 🌟 수정 2: 'all' 모드일 때도 예측이 실행되도록 'in ("predict", "all")' 로 변경합니다.
    # if args.mode in ("predict", "all"):
    if args.mode == "predict":
        
        # 혹시라도 경로가 비어있을 경우를 대비한 안전장치
        if not args.source:
            args.source = '/content/data/v3/test_images' 
            
        # from model import predict
        print("\n=== 예측 ===")
        t2.predict(args.source, weights=args.weights, conf=args.conf)

if __name__ == "__main__":
    main()