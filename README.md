# LCMD-Det
Relevant source will be released to this GitHub repo upon acceptance.


## 1. Detailed information
- The **deployment** scripts and settings are in [`deployment`](./deployment).  
- The experiment setting configs are in [`configs`](./configs).  


## 2. Data and trained weights
Relevant datasets and trained model checkpoints can be accessed from our [Google Drive]().

## 3. Implement
### 3.1 Environment Setup
```
pip install -r requirements.txt
```

### 3.2 Train
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


### 3.3 Val
```
python val.py --save true
```

### 3.4 Heatmap generation
```
python heatmap.py
```
