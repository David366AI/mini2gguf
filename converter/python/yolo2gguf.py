from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from converter.python.onnx2gguf import (
	_load_onnx_model_for_ggml_with_warnings,
	export_ir_json_from_model,
	export_weights_gguf_from_model,
	resolve_output_paths,
)
from converter.python.pt_exporters import export_pt_to_onnx


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="yolo2gguf.py",
		description="Convert YOLO/CRNN PT or ONNX model to mini2gguf graph JSON + GGUF weights",
	)
	parser.add_argument("-i", "--input", required=True, help="Input model path (.pt or .onnx)")
	parser.add_argument(
		"-o",
		"--output-dir",
		default=None,
		help="Output directory (default: same directory as ONNX model)",
	)
	parser.add_argument("-v", "--model-version", required=True, help="Model version, e.g. 5/6/7/8/11/26")
	parser.add_argument(
		"-c",
		"--model-class",
		required=True,
		choices=["detection", "segmentation", "classification", "pose", "obb"],
		help="Model class/task type",
	)
	parser.add_argument(
		"--export-backend",
		default="auto",
		choices=["auto", "ultralytics", "yolov5"],
		help="PT->ONNX export backend (default: auto)",
	)
	parser.add_argument(
		"--yolov5-export-script",
		default=None,
		help="Optional path to external yolov5 export.py (e.g. /mnt/c/ocr/yolov5-7.0/export.py)",
	)
	parser.add_argument(
		"--yolov5-python",
		default=None,
		help="Optional Python executable used to run yolov5 export.py",
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
	return parser


def _resolve_input_onnx(
	input_path: Path,
	export_backend: str,
	yolov5_export_script: str | None,
	yolov5_python: str | None,
	export_half: bool,
) -> tuple[Path, bool, str | None]:
	suffix = input_path.suffix.lower()
	if suffix == ".onnx":
		return input_path, False, None
	if suffix == ".pt":
		onnx_path, backend_name = export_pt_to_onnx(
			input_path,
			PROJECT_ROOT,
			backend=export_backend,
			yolov5_export_script=yolov5_export_script,
			yolov5_python=yolov5_python,
			export_half=export_half,
		)
		return onnx_path, True, backend_name
	raise ValueError(f"Unsupported input format: {input_path.suffix}. Only .pt and .onnx are supported")


def main() -> int:
	args = _build_parser().parse_args()
	input_path = Path(args.input).expanduser().resolve()
	if not input_path.exists():
		raise FileNotFoundError(f"Input model not found: {input_path}")

	onnx_path, should_delete_onnx, used_export_backend = _resolve_input_onnx(
		input_path,
		args.export_backend,
		args.yolov5_export_script,
		args.yolov5_python,
		args.export_half,
	)
	try:
		output_dir = args.output_dir if args.output_dir is not None else str(input_path.parent)
		graph_output, weights_output = resolve_output_paths(onnx_path, output_dir)

		model, rewrite_warnings = _load_onnx_model_for_ggml_with_warnings(onnx_path)
		graph_path = export_ir_json_from_model(
			model=model,
			output_path=graph_output,
			include_initializer_values=False,
			indent=2,
		)

		extra_metadata = {
			"model.family": "yolo",
			"model.version": str(args.model_version),
			"model.class": args.model_class,
		}
		weights_path = export_weights_gguf_from_model(
			model=model,
			weights_output_path=weights_output,
			extra_metadata=extra_metadata,
			cast_weight_to_f16=args.weight_f16,
		)

		for warning in rewrite_warnings:
			print(f"WARNING: {warning}")

		if input_path.suffix.lower() == ".pt":
			if used_export_backend is not None:
				print(f"PT export backend: {used_export_backend}")
			print(f"PT exported to ONNX: {onnx_path}")
		print(f"Graph JSON exported: {graph_path}")
		print(f"Weights GGUF exported: {weights_path}")
		print(
			"GGUF metadata written: "
			f"model.family={extra_metadata['model.family']}, "
			f"model.version={extra_metadata['model.version']}, "
			f"model.class={extra_metadata['model.class']}"
		)
	finally:
		# if should_delete_onnx and onnx_path.exists():
		# 	onnx_path.unlink()
		# 	print(f"Temporary ONNX removed: {onnx_path}")
		pass
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
