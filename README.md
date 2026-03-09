# mini2gguf
mini2gguf is an open-source toolkit that converts compact AI models (PyTorch/ONNX) into GGUF weights plus a portable graph JSON, then runs dynamic inference on top of ggml. It focuses on lightweight vision workloads and provides a C/C++ runtime API for model initialization, multithreaded inference, optional post-processing, and executable examples.

## Python IR (ONNX -> graph.json + weights.gguf)

Use the Python converter to generate both files in one pass:

- `graph.json`: graph structure and tensor metadata
- `weights.gguf`: initializer weights for runtime loading

CLI output naming:

- Input `-i path/to/blank_16.onnx`
- Output files: `blank_16_graph.json` and `blank_16_weights.gguf`
- `-o` not set: outputs go to the same directory as input
- `-o <dir>` set: outputs go to that directory

ggml compatibility in converter:

- Rank-5 tensors are lowered to rank-4 by merging `H x W`
- Related `Transpose` perms and axis-based attributes are rewritten accordingly
- Rank `>= 6` tensors are rejected with an error

By default, `graph.json` is metadata-only (no initializer values), so weights stay in GGUF.

```bash
conda run -n base python -m converter.python.onnx_to_gguf \
	-i assets/models/yolo/raw/blank_16.onnx \
	-o assets/models/yolo/converted
```

This command writes:

- `assets/models/yolo/converted/blank_16_graph.json`
- `assets/models/yolo/converted/blank_16_weights.gguf`

If `-o` is omitted:

```bash
conda run -n base python -m converter.python.onnx_to_gguf \
	-i assets/models/yolo/raw/blank_16.onnx
```

Then outputs are created under `assets/models/yolo/raw/` with the same naming rule.

Run unit tests:

```bash
conda run -n base python -m unittest discover -s tests/unit -p "test_*.py"
```

## C++ Runtime (Dynamic Graph)

Runtime does not hardcode class count or YOLO-specific output width. It uses `graph.json` tensor metadata at load time, so different ONNX-derived models can have different output shapes.

Build with local `ggml/` clone:

```bash
cmake -S . -B build
cmake --build build -j
```

Run demo (loads `<model_name>_graph.json` + `<model_name>_weights.gguf`):

```bash
./build/examples/cpp/mini2gguf_infer_demo assets/models/yolo/converted blank_16
```

CPU runtime notes:

- CPU direct conv path is enabled by default when available.
- For CPU direct conv, runtime auto-skips `graph_input_cast0` (`F32->F16`) to avoid redundant `F32<->F16` conversion on the hot path.
- Set `MINI2GGUF_DISABLE_AUTO_SKIP_INPUT_F16_CAST=1` to restore strict graph Cast behavior.
