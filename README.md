# LCMD-Det

Official implementation of **"Infrared-assisted Cross-Modality Detection for Construction Site Worker Safety Monitoring"**.

## 1. Implementation

### 1.1 Environment Setup

```bash
pip install -r requirements.txt
```

### 1.2 Model Configs

Model configuration files are provided in:

```bash
./model_configs/
```

### 1.3 Model Weights

Pre-trained model weights are provided in:

```bash
./weights/
```

### 1.4 Train

```bash
python train.py
```

Key arguments are configured inside `train.py`:

| Argument  | Description                    |
| --------- | ------------------------------ |
| `data`    | Path to dataset `.yaml` config |
| `imgsz`   | Input image size, default: 640 |
| `epochs`  | Number of training epochs      |
| `batch`   | Batch size                     |
| `device`  | GPU device id(s)               |
| `project` | Output directory for runs      |

### 1.5 Val

```bash
python val.py
```

## 2. Dataset

The dataset can be downloaded from [Google Drive]([https://github.com/Hongru0306/LCMD-Det/edit/main/README.md](https://drive.google.com/file/d/1JJkr1q4J8Uk1NhYoFeVC3L-luynv8Vc9/view?usp=sharing)).


## 3. Deployment

Deployment scripts and settings are provided in:

```bash
./deployment/
```
