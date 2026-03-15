from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from converter.onnx2gguf import (
    _load_onnx_model_for_ggml_with_warnings,
    export_weights_gguf_from_model,
    graph_json_text_from_model,
    resolve_output_paths,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crnn2gguf.py",
        description="Convert CRNN ONNX model to mini2gguf GGUF weights (graph is embedded by default)",
    )
    parser.add_argument("-i", "--input", required=True, help="Input model path (.onnx)")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory (default: same directory as ONNX model)",
    )
    parser.add_argument(
        "-d",
        "--dict",
        required=True,
        help="Dictionary file path (one character per line) for CTC decode metadata",
    )
    parser.add_argument(
        "--export-half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When input is PT, export ONNX in float16 (default: enabled). Use --no-export-half for float32 ONNX.",
    )
    parser.add_argument(
        "--weight-f16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cast floating-point weights to float16 when writing GGUF (default: enabled).",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Also export standalone <stem>.json and do not embed graph JSON into GGUF metadata.",
    )
    return parser


def _load_dict_lines(dict_path: Path) -> list[str]:
    if not dict_path.exists() or not dict_path.is_file():
        raise FileNotFoundError(f"Dictionary file not found: {dict_path}")

    lines: list[str] = []
    with dict_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.endswith("\r"):
                line = line[:-1]
            lines.append(line)

    if not lines:
        raise ValueError(f"Dictionary file is empty: {dict_path}")
    return lines


def main() -> int:
    args = _build_parser().parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input model not found: {input_path}")
    if input_path.suffix.lower() != ".onnx":
        raise ValueError(
            f"Unsupported input format: {input_path.suffix}. crnn2gguf currently supports .onnx only"
        )

    dict_path = Path(args.dict).expanduser().resolve()
    dict_lines = _load_dict_lines(dict_path)

    output_dir = args.output_dir if args.output_dir is not None else str(input_path.parent)
    graph_output, weights_output = resolve_output_paths(input_path, output_dir)

    model, rewrite_warnings = _load_onnx_model_for_ggml_with_warnings(input_path)
    graph_json = graph_json_text_from_model(
        model=model,
        include_initializer_values=False,
        indent=2,
    )

    graph_path: Path | None = None
    if args.split:
        graph_path = graph_output
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_text(graph_json, encoding="utf-8")

    extra_metadata = {
        "model.family": "crnn",
        "model.dict": "".join(dict_lines),
    }
    weights_path = export_weights_gguf_from_model(
        model=model,
        weights_output_path=weights_output,
        extra_metadata=extra_metadata,
        cast_weight_to_f16=args.weight_f16,
        embedded_graph_json=None if args.split else graph_json,
    )

    for warning in rewrite_warnings:
        print(f"WARNING: {warning}")

    if graph_path is not None:
        print(f"Graph JSON exported: {graph_path}")
    else:
        print("Graph JSON embedded into GGUF metadata key: model.graph")
    print(f"Weights GGUF exported: {weights_path}")
    graph_meta_state = "external(--split)" if args.split else "embedded"
    print(
        "GGUF metadata written: "
        f"model.family={extra_metadata['model.family']}, "
        f"model.dict=<lines:{len(dict_lines)}>, "
        f"model.graph={graph_meta_state}"
    )
    if args.input.lower().endswith(".onnx") and not args.export_half:
        print("NOTE: --no-export-half has no effect for ONNX input.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
