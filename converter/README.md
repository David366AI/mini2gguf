# Converter Guide

This directory contains conversion tools for model formats used by `mini2gguf`.

All examples below assume you run commands from the project root:

```bash
cd /path/to/mini2gguf
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

For YOLOv5 PT export backend support, see [third_party/README.md](../third_party/README.md).

## 1) `darknet2onnx.py`

Convert Darknet YOLO (`.cfg` + `.weights`) to ONNX.

Script:

```bash
python converter/darknet2onnx.py --cfg <model.cfg> --weights <model.weights> [--output-path <out.onnx>]
```

Arguments:

- `-c, --cfg` (required): Darknet `.cfg` path
- `-w, --weights` (required): Darknet `.weights` path
- `-o, --output-path` (optional): output ONNX path, default is `<cfg_dir>/<cfg_stem>.onnx`

Example:

```bash
python converter/darknet2onnx.py \
  --cfg assets/models/yolo/yolov4-tiny.cfg \
  --weights assets/models/yolo/yolov4-tiny.weights \
  --output-path assets/models/yolo/yolov4-tiny.onnx
```

Notes:

- This script writes ONNX metadata such as:
  - `mini2gguf.darknet.yolo`
  - `model.family=yolo`
  - `model.version=3`
  - `model.class=detection`

## 2) `onnx2gguf.py`

Convert ONNX to GGUF (weights), and embed graph JSON in GGUF metadata by default.

Script:

```bash
python converter/onnx2gguf.py -i <model.onnx> [options]
```

Arguments:

- `-i, --input` (required): input ONNX
- `-o, --output-dir`: output directory (default: ONNX directory)
- `--indent`: JSON indent for graph serialization
- `--weight-f16/--no-weight-f16`: cast weight tensors to fp16 (default: enabled)
- `--split`: export standalone `<stem>.json` and do not embed `model.graph` into GGUF
- `--model-family`: optional metadata value, e.g. `yolo`, `crnn`

Examples:

```bash
# Generic ONNX -> GGUF, with graph embedded into model.graph
python converter/onnx2gguf.py -i assets/models/yolo/yolov5n.onnx

# For CRNN or other families, set model.family explicitly
python converter/onnx2gguf.py -i assets/models/crnn/crnn.onnx --model-family crnn

# Split graph json out for debugging
python converter/onnx2gguf.py -i assets/models/yolo/yolov5n.onnx --split
```

## 3) `yolo2gguf.py`

One-step converter for YOLO models (`.pt` or `.onnx`) to GGUF.

Script:

```bash
python converter/yolo2gguf.py -i <model.pt|model.onnx> -v <version> -c <class> [options]
```

Arguments:

- `-i, --input` (required): `.pt` or `.onnx`
- `-v, --model-version` (required): e.g. `5`, `8`, `11`, `26`
- `-c, --model-class` (required): one of `detection|segmentation|classification|pose|obb`
- `-o, --output-dir`: output directory
- `--export-backend`: `auto|ultralytics|yolov5` (PT input only)
- `--yolov5-dir`: YOLOv5 repo directory (if omitted, auto-search `./yolov5`, `<repo>/third_party/yolov5`, `../yolov5`)
- `--yolov5-python`: Python executable for YOLOv5 export
- `--export-half/--no-export-half`: PT->ONNX export precision toggle
- `--weight-f16/--no-weight-f16`: GGUF weight cast toggle
- `--split`: export `<stem>.json` separately instead of embedding graph JSON
- `-keep-onnx, --keep-onnx`: keep intermediate ONNX files for PT input (default behavior removes them after successful conversion)

Examples:

```bash
# YOLOv5 PT -> GGUF (use YOLOv5 backend)
python converter/yolo2gguf.py \
  -i assets/models/yolo/yolov5n.pt \
  -v 5 \
  -c detection \
  --export-backend yolov5 \
  --yolov5-dir ./third_party/yolov5

# YOLOv8/11/26 PT -> GGUF (Ultralytics backend, usually auto)
python converter/yolo2gguf.py \
  -i assets/models/yolo/yolo26n.pt \
  -v 26 \
  -c detection

# ONNX input is also supported
python converter/yolo2gguf.py \
  -i assets/models/yolo/yolov5n.onnx \
  -v 5 \
  -c detection
```

Notes:

- `yolo2gguf.py` always writes YOLO metadata:
  - `model.family=yolo`
  - `model.version=<your -v>`
  - `model.class=<your -c>`
- When input is `.pt`, intermediate `.onnx` and `.onnx.data` are removed after successful conversion.

## End-to-End: Darknet -> ONNX -> GGUF

```bash
python converter/darknet2onnx.py \
  --cfg assets/models/yolo/yolov4-tiny.cfg \
  --weights assets/models/yolo/yolov4-tiny.weights \
  --output-path assets/models/yolo/yolov4-tiny.onnx

python converter/onnx2gguf.py \
  -i assets/models/yolo/yolov4-tiny.onnx \
  --model-family yolo
```
