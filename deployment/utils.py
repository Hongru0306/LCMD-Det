from __future__ import annotations

import json
import statistics
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from prettytable import PrettyTable
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_gradients, get_num_params, select_device, time_sync

from deployment.config import DEFAULT_IMGSZ


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def benchmark_latency(
    model: YOLO,
    image_path: Path,
    imgsz: int = DEFAULT_IMGSZ,
    device: str = "0",
    warmup: int = 3,
    iters: int = 10,
    half: bool = False,
) -> dict[str, Any]:
    """Measure end-to-end single-image latency using YOLO.predict."""
    latencies = []
    predict_kwargs = {
        "source": str(image_path),
        "imgsz": imgsz,
        "device": device,
        "verbose": False,
        "half": half,
    }

    for step in range(warmup + iters):
        start = time_sync()
        model.predict(**predict_kwargs)
        latency_ms = (time_sync() - start) * 1000
        if step >= warmup:
            latencies.append(latency_ms)

    mean_latency = statistics.mean(latencies) if latencies else None
    std_latency = statistics.pstdev(latencies) if len(latencies) > 1 else 0.0
    return {
        "warmup": warmup,
        "iters": iters,
        "latencies_ms": latencies,
        "mean_latency_ms": mean_latency,
        "std_latency_ms": std_latency,
        "fps": (1000.0 / mean_latency) if mean_latency else None,
    }


def collect_pt_model_profile(model: YOLO, imgsz: int = DEFAULT_IMGSZ) -> dict[str, Any]:
    """Collect parameter and FLOPs info for a PyTorch model."""
    torch_model = deepcopy(model.model).float().eval()
    params = get_num_params(torch_model)
    grads = get_num_gradients(torch_model)
    flops = get_flops(torch_model, imgsz=imgsz)
    return {
        "layers": len(list(torch_model.modules())),
        "parameters": params,
        "gradients": grads,
        "flops_g": flops,
    }


def collect_runtime_context(device: str) -> dict[str, Any]:
    torch_device = select_device(device, verbose=False)
    context = {
        "device": str(torch_device),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available() and torch_device.type == "cuda":
        props = torch.cuda.get_device_properties(torch_device)
        context.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_mb": round(props.total_memory / 1024 / 1024, 2),
                "cuda_device_index": torch_device.index,
            }
        )
    return context


def collect_val_metrics(result: Any) -> dict[str, Any]:
    metrics = {k: float(v) for k, v in result.results_dict.items()}
    speed = {k: float(v) for k, v in result.speed.items()}
    metrics["speed"] = speed
    metrics["fps_inference_only"] = 1000.0 / speed["inference"] if speed.get("inference") else None
    total = speed.get("preprocess", 0.0) + speed.get("inference", 0.0) + speed.get("postprocess", 0.0)
    metrics["fps_end_to_end"] = 1000.0 / total if total else None
    return metrics


def profile_to_table(report: dict[str, Any]) -> str:
    table = PrettyTable()
    table.title = "Deployment Summary"
    table.field_names = ["Item", "PyTorch", "TensorRT"]

    pt = report["pytorch"]
    trt = report["tensorrt"]
    table.add_row(["Model Path", pt["model_path"], trt["model_path"]])
    table.add_row(["File Size (MB)", format_float(pt["file_size_mb"]), format_float(trt["file_size_mb"])])
    table.add_row(["Val mAP50-95", format_float(pt["val_metrics"].get("metrics/mAP50-95(B)"), 4), format_float(trt["val_metrics"].get("metrics/mAP50-95(B)"), 4)])
    table.add_row(["Val inference ms/img", format_float(pt["val_metrics"]["speed"].get("inference")), format_float(trt["val_metrics"]["speed"].get("inference"))])
    table.add_row(["Val FPS (inference)", format_float(pt["val_metrics"].get("fps_inference_only"), 2), format_float(trt["val_metrics"].get("fps_inference_only"), 2)])
    table.add_row(["Latency ms/img", format_float(pt["latency"].get("mean_latency_ms")), format_float(trt["latency"].get("mean_latency_ms"))])
    table.add_row(["Latency FPS", format_float(pt["latency"].get("fps"), 2), format_float(trt["latency"].get("fps"), 2)])
    return table.get_string()


def save_json_report(report: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
