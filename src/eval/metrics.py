def normalize_yolo_metrics(results_dict: dict) -> dict:
    output = {
        "mAP50": None,
        "mAP50_95": None,
        "precision": None,
        "recall": None,
    }

    key_map = {
        "metrics/mAP50(B)": "mAP50",
        "metrics/mAP50-95(B)": "mAP50_95",
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
    }

    for src_key, dst_key in key_map.items():
        if src_key in results_dict:
            try:
                output[dst_key] = float(results_dict[src_key])
            except Exception:
                output[dst_key] = results_dict[src_key]

    return output