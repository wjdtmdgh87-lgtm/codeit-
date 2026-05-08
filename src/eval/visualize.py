import pandas as pd
import matplotlib.pyplot as plt


def plot_yolo_comparison(csv_path, save_path=None):
    df = pd.read_csv(csv_path)

    if "experiment_name" not in df.columns or "mAP50_95" not in df.columns:
        raise ValueError("CSV must contain experiment_name and mAP50_95 columns")

    df = df.sort_values("mAP50_95", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(df["experiment_name"], df["mAP50_95"])
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("mAP50-95")
    plt.title("YOLO Experiment Comparison")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()