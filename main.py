"""
[수정 사항]
- 기존 data / train / predict / all 모드는 유지함
- experiment 모드를 추가해서 실험 이름별로 학습을 실행할 수 있게 함
- experiment_all 모드를 추가해서 여러 실험을 순차 실행할 수 있게 함
- compare 모드를 추가해서 누적된 실험 결과를 비교할 수 있게 함
- 기존 한국어 사용법/출력문은 유지함

사용법:
  python main.py --mode data
  python main.py --mode train
  python main.py --mode predict --source data/images/test/
  python main.py --mode all
  python main.py --mode experiment --exp baseline
  python main.py --mode experiment_all
  python main.py --mode compare
"""

import sys
import argparse
from pathlib import Path

# src/ 를 모듈 경로에 추가 (main.py 는 project_root/ 에서 실행)
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["data", "train", "predict", "all", "experiment", "experiment_all", "compare"],
        default="all"
    )
    parser.add_argument("--source", default=None, help="predict 모드 시 이미지 경로")
    parser.add_argument("--weights", default=None, help="predict 모드 시 가중치 경로")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--exp", default="baseline", help="experiment 모드 실험 이름")
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

    if args.mode == "experiment":
        from experiment import run_experiment
        print("\n=== 실험 실행 ===")
        run_experiment(args.exp)

    if args.mode == "experiment_all":
        from experiment import run_all_experiments
        print("\n=== 전체 실험 실행 ===")
        run_all_experiments()

    if args.mode == "compare":
        from experiment import compare_results
        print("\n=== 실험 비교 ===")
        compare_results()


if __name__ == "__main__":
    main()