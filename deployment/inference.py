from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from deployment.config import (
    DATA_YAML,
    DEFAULT_BATCH,
    DEFAULT_DEVICE,
    DEFAULT_IMGSZ,
    ENGINE_MODEL_PATH,
    OUTPUT_DIR,
    PT_MODEL_PATH,
    SAMPLE_IMAGE,
)
from deployment.utils import (
    benchmark_latency,
    collect_pt_model_profile,
    collect_runtime_context,
    collect_val_metrics,
    ensure_dir,
    file_size_mb,
    profile_to_table,
    save_json_report,
)
from ultralytics import YOLO


def build_runtime_report(model_path, model_label: str, is_pytorch: bool):
    model = YOLO(str(model_path))
    val_result = model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=DEFAULT_IMGSZ,
        batch=DEFAULT_BATCH,
        device=DEFAULT_DEVICE,
        save_json=False,
        project=str(OUTPUT_DIR / "val_runs"),
        name=model_label,
        exist_ok=True,
        plots=False,
        verbose=True,
    )

    latency = benchmark_latency(
        model=model,
        image_path=SAMPLE_IMAGE,
        imgsz=DEFAULT_IMGSZ,
        device=DEFAULT_DEVICE,
        warmup=3,
        iters=10,
        half=False,
    )

    static_profile = collect_pt_model_profile(model, imgsz=DEFAULT_IMGSZ) if is_pytorch else {}

    return {
        "model_label": model_label,
        "model_path": str(model_path),
        "file_size_mb": file_size_mb(model_path),
        "static_profile": static_profile,
        "val_metrics": collect_val_metrics(val_result),
        "latency": latency,
    }


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    report = {
        "runtime": collect_runtime_context(DEFAULT_DEVICE),
        "settings": {
            "imgsz": DEFAULT_IMGSZ,
            "batch": DEFAULT_BATCH,
            "device": DEFAULT_DEVICE,
            "data": str(DATA_YAML),
        },
        "pytorch": build_runtime_report(PT_MODEL_PATH, "pytorch", is_pytorch=True),
        "tensorrt": build_runtime_report(ENGINE_MODEL_PATH, "tensorrt", is_pytorch=False),
    }

    save_json_report(report, OUTPUT_DIR / "deployment_report.json")
    summary = profile_to_table(report)
    (OUTPUT_DIR / "deployment_summary.txt").write_text(summary + "\n", encoding="utf-8")

    print(summary)
    print(f"\nDetailed report saved to: {OUTPUT_DIR / 'deployment_report.json'}")
    print(f"Summary table saved to : {OUTPUT_DIR / 'deployment_summary.txt'}")


if __name__ == "__main__":
    main()
