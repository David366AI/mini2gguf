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
        yolov5_dir: str | None = None,
        python_executable: str | None = None,
        **kwargs: object,
    ) -> Path:
        export_half = bool(kwargs.get("export_half", True))
        export_script = None
        if yolov5_dir:
            input_path = Path(yolov5_dir).expanduser().resolve()
            if input_path.is_dir():
                export_script = input_path / "export.py"
            elif input_path.is_file() and input_path.name == "export.py":
                # Backward-compatible path support: user may still pass export.py directly.
                export_script = input_path
            else:
                raise RuntimeError(
                    "Invalid --yolov5-dir path. It must be a YOLOv5 repository directory "
                    f"(or a direct export.py path for compatibility): {input_path}"
                )
            if not export_script.exists():
                raise RuntimeError(
                    f"YOLOv5 installation not found under --yolov5-dir: {input_path}. "
                    "Expected: <yolov5_dir>/export.py"
                )
        else:
            export_script = _find_yolov5_export_script(project_root)
        if export_script is None:
            raise RuntimeError(
                "YOLOv5 export backend selected, but YOLOv5 installation was not found. "
                "Expected one of: ./yolov5, <repo>/third_party/yolov5, ../yolov5 (each containing export.py).\n"
                "If you are converting YOLOv5 PT (e.g. -v 5), install YOLOv5 first (see third_party/README.md),\n"
                "or pass YOLOv5 repo directory via --yolov5-dir.\n"
                "You can also set the Python executable via --yolov5-python."
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
        if proc.returncode != 0 and export_half:
            half_cpu_err = (
                "--half only compatible with GPU export" in proc.stderr
                or "--half only compatible with GPU export" in proc.stdout
            )
            if half_cpu_err:
                retry_cmd = [arg for arg in cmd if arg != "--half"]
                print(
                    "WARNING: yolov5 export with --half failed on CPU; retrying without --half."
                )
                proc = subprocess.run(retry_cmd, capture_output=True, text=True)
                cmd = retry_cmd

        if proc.returncode != 0:
            raise RuntimeError(
                "YOLOv5 PT->ONNX conversion failed.\n"
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
            "YOLOv5 PT->ONNX conversion finished but ONNX file was not found near input PT. "
            f"Searched for: {direct} and {input_pt.stem}*.onnx in {input_pt.parent}"
        )


def export_pt_to_onnx(
    input_pt: Path,
    project_root: Path,
    backend: str = "auto",
    yolov5_dir: str | None = None,
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
            yolov5_dir=yolov5_dir,
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
        yolov5_dir=yolov5_dir,
        python_executable=yolov5_python,
        export_half=export_half,
    ), v5.backend_name
