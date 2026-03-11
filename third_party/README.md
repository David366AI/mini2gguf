# Third-Party Dependencies

## YOLOv5 Backend (for `yolo2gguf --export-backend yolov5`)

This project can use the official YOLOv5 `export.py` as a PT -> ONNX backend.

### 1. Clone YOLOv5

```bash
git clone https://github.com/ultralytics/yolov5.git third_party/yolov5
```

### 2. Install YOLOv5 Dependencies

```bash
pip install -r third_party/yolov5/requirements.txt
```

Note:
- This path requires `torch` (via YOLOv5/Ultralytics export flow).

### 3. Download a Sample YOLOv5 Model

```bash
cd assets/models/yolo
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt
```

### 4. Convert PT -> GGUF with YOLOv5 Export Backend

Run from the project root:

```bash
python converter/yolo2gguf.py \
  -i assets/models/yolo/yolov5n.pt \
  -v 5 \
  -c detection \
  --yolov5-dir ./third_party/yolov5
```
