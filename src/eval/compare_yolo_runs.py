import pandas as pd
from config.settings import YOLO_OUTPUT_DIR


def main():
    csv_path = YOLO_OUTPUT_DIR / "yolo_experiment_summary.csv"
    df = pd.read_csv(csv_path)

    print("\n=== All YOLO Experiments ===")
    print(df)

    print("\n=== Top 5 by mAP50 ===")
    if "mAP50" in df.columns:
        print(df.sort_values("mAP50", ascending=False).head(5))

    print("\n=== Top 5 by mAP50_95 ===")
    if "mAP50_95" in df.columns:
        print(df.sort_values("mAP50_95", ascending=False).head(5))

    print("\n=== Top 5 by Precision ===")
    if "precision" in df.columns:
        print(df.sort_values("precision", ascending=False).head(5))

    print("\n=== Top 5 by Recall ===")
    if "recall" in df.columns:
        print(df.sort_values("recall", ascending=False).head(5))


if __name__ == "__main__":
    main()