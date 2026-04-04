# Deployment

> Checkpoints will be released after acceptance.

This repository is currently in anonymous/review mode.
The code for training, validation, export, and deployment benchmarking is provided for reproducibility review.
Pretrained checkpoints are intentionally not included at this stage.

## What's available now

- TensorRT export script in `deployment/tools.py`
- PyTorch vs TensorRT validation/latency comparison in `deployment/inference.py`
- Deployment reports saved to `deployment/outputs/`

## Deployment workflow

### 1. Prepare a checkpoint

Update `deployment/config.py` so that `PT_MODEL_PATH` points to your trained `.pt` file.
The TensorRT engine path is derived automatically from the same run directory.

### 2. Export TensorRT engine

```bash
python deployment/tools.py
```

This will:
- load the PyTorch checkpoint
- export a TensorRT engine
- save an export report to `deployment/outputs/export_report.json`

### 3. Run validation and benchmarking

```bash
python deployment/inference.py
```

This will compare PyTorch and TensorRT on:
- file size
- validation mAP
- validation inference speed
- single-image latency
- FPS

Outputs:
- `deployment/outputs/deployment_report.json`
- `deployment/outputs/deployment_summary.txt`

## Notes

- For fair comparison, PyTorch and TensorRT are evaluated with the same dataset and image size settings from `deployment/config.py`.
- GPU benchmarking in our experiments was run with CUDA-enabled PyTorch and TensorRT.

## Review statement

A full project README with checkpoint links, training recipes, and citation information will be added after acceptance.
