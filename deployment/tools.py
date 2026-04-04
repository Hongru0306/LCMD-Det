from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from deployment.config import (
    DEFAULT_DEVICE,
    DEFAULT_IMGSZ,
    ENGINE_MODEL_PATH,
    OUTPUT_DIR,
    PT_MODEL_PATH,
)
from deployment.utils import ensure_dir, file_size_mb, save_json_report
from ultralytics import YOLO


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    model = YOLO(str(PT_MODEL_PATH))
    export_path = model.export(format="engine", device=DEFAULT_DEVICE, imgsz=DEFAULT_IMGSZ)

    report = {
        "source_model": str(PT_MODEL_PATH),
        "exported_model": str(export_path),
        "expected_engine_path": str(ENGINE_MODEL_PATH),
        "source_size_mb": file_size_mb(PT_MODEL_PATH),
        "engine_size_mb": file_size_mb(export_path),
        "imgsz": DEFAULT_IMGSZ,
        "device": DEFAULT_DEVICE,
    }
    save_json_report(report, OUTPUT_DIR / "export_report.json")
    print("TensorRT export completed.")
    print(f"PyTorch weights : {PT_MODEL_PATH}")
    print(f"TensorRT engine : {export_path}")
    print(f"Artifacts saved : {OUTPUT_DIR / 'export_report.json'}")


if __name__ == "__main__":
    main()
