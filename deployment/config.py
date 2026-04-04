from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" /  "exp8"
WEIGHTS_DIR = RUN_DIR / "weights"

PT_MODEL_PATH = WEIGHTS_DIR / "best.pt"
ENGINE_MODEL_PATH = WEIGHTS_DIR / "best.engine"
DATA_YAML = Path("./data/M3FD_day_night.yaml")
VAL_IMAGE_DIR = Path("./images/val")
SAMPLE_IMAGE = VAL_IMAGE_DIR / "00001.png"
OUTPUT_DIR = ROOT / "deployment" / "artifacts"
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_DEVICE = "0"
