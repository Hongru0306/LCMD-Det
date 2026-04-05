# LCMD-Det
Relevant source will be released to this GitHub repo upon acceptance.


## 1. Deployment details
- The **deployment** scripts and settings are in [`deployment`](./deployment).   


## 2. Implement
### 2.1 Environment Setup
```
pip install -r requirements.txt
```

### 2.2 Train
```
python train.py
```

Key arguments (configured inside `train.py`):

| Argument | Description |
|---|---|
| `data` | Path to dataset `.yaml` config |
| `imgsz` | Input image size (default: 640) |
| `epochs` | Number of training epochs |
| `batch` | Batch size |
| `device` | GPU device id(s) |
| `project` | Output directory for runs |


### 2.3 Val
```
python val.py --save true
```

### 2.4 Heatmap generation
```
python heatmap.py
```
