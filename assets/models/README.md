# Model Assets

This folder stores local test models and converted artifacts for `mini2gguf`.

## Structure

- `yolo/raw/`: original model files (`.pt`, `.onnx`)
- `yolo/converted/`: exported files (`.gguf`, `graph.json`, debug dumps)
- `crnn/raw/`: original model files (`.pt`, `.onnx`)
- `crnn/converted/`: exported files (`.gguf`, `graph.json`, debug dumps)

## Notes

- Large model binaries are ignored by default via `.gitignore`.
- Keep only tiny sample files if you want to commit demo assets.
- Record model source, license, and preprocessing config in this folder when needed.
