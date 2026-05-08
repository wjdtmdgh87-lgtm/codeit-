from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
YOLO_OUTPUT_DIR = OUTPUT_DIR / "yolo"

for path in [OUTPUT_DIR, YOLO_OUTPUT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42