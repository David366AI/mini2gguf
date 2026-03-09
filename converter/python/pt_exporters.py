from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _find_yolov5_export_script(project_root: Path) -> Path | None:
    candidates = [
        Path.cwd() / "yolov5" / "export.py",
        project_root / "third_party" / "yolov5" / "export.py",
        project_root.parent / "yolov5" / "export.py",
    ]
    for script in candidates:
        if script.exists():
            return script.resolve()
    return None


class PTExporter:
    backend_name: str = "unknown"

    def export(self, input_pt: Path, project_root: Path, **kwargs: object) -> Path:
        raise NotImplementedError


class UltralyticsExporter(PTExporter):
    backend_name = "ultralytics"

    def export(self, input_pt: Path, project_root: Path, **kwargs: object) -> Path:
        export_half = bool(kwargs.get("export_half", True))
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "PT->ONNX export via ultralytics requires package 'ultralytics' in current environment."
            ) from exc

        model = YOLO(str(input_pt))
        exported = model.export(format="onnx", half=export_half)
        onnx_path = Path(exported)
        if not onnx_path.exists():
            raise RuntimeError(f"Ultralytics export did not produce ONNX file: {onnx_path}")
        return onnx_path.resolve()


class Yolov5Exporter(PTExporter):
    backend_name = "yolov5"

    def export(
        self,
        input_pt: Path,
        project_root: Path,
        yolov5_export_script: str | None = None,
        python_executable: str | None = None,
        **kwargs: object,
    ) -> Path:
        export_half = bool(kwargs.get("export_half", True))
        export_script = None
        if yolov5_export_script:
            export_script = Path(yolov5_export_script).expanduser().resolve()
            if not export_script.exists():
                raise RuntimeError(f"yolov5 export script not found: {export_script}")
        else:
            export_script = _find_yolov5_export_script(project_root)
        if export_script is None:
            raise RuntimeError(
                "YOLOv5 export backend selected, but yolov5/export.py was not found. "
                "Expected one of: ./yolov5/export.py, <repo>/third_party/yolov5/export.py, ../yolov5/export.py"
            )

        py_exec = python_executable if python_executable else sys.executable

        cmd = [
            py_exec,
            str(export_script),
            "--weights",
            str(input_pt),
            "--include",
            "onnx",
        ]
        if export_half:
            cmd.append("--half")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "YOLOv5 export.py failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )

        direct = input_pt.with_suffix(".onnx")
        if direct.exists():
            return direct.resolve()

        nearby = sorted(input_pt.parent.glob(f"{input_pt.stem}*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
        if nearby:
            return nearby[0].resolve()

        raise RuntimeError(
            "YOLOv5 export.py finished but ONNX file was not found near input PT. "
            f"Searched for: {direct} and {input_pt.stem}*.onnx in {input_pt.parent}"
        )


def export_pt_to_onnx(
    input_pt: Path,
    project_root: Path,
    backend: str = "auto",
    yolov5_export_script: str | None = None,
    yolov5_python: str | None = None,
    export_half: bool = True,
) -> tuple[Path, str]:
    backend = backend.lower().strip()
    if backend not in {"auto", "ultralytics", "yolov5"}:
        raise ValueError(f"Unsupported export backend: {backend}. Use one of auto/ultralytics/yolov5")

    if backend == "ultralytics":
        exporter = UltralyticsExporter()
        return exporter.export(input_pt, project_root, export_half=export_half), exporter.backend_name

    if backend == "yolov5":
        exporter = Yolov5Exporter()
        return exporter.export(
            input_pt,
            project_root,
            yolov5_export_script=yolov5_export_script,
            python_executable=yolov5_python,
            export_half=export_half,
        ), exporter.backend_name

    ultra = UltralyticsExporter()
    try:
        return ultra.export(input_pt, project_root, export_half=export_half), ultra.backend_name
    except Exception as exc:
        message = str(exc)
        is_yolov5_ckpt = (
            "appears to be an Ultralytics YOLOv5 model" in message
            or "No module named 'models'" in message
        )
        if not is_yolov5_ckpt:
            raise

    v5 = Yolov5Exporter()
    return v5.export(
        input_pt,
        project_root,
        yolov5_export_script=yolov5_export_script,
        python_executable=yolov5_python,
        export_half=export_half,
    ), v5.backend_name
