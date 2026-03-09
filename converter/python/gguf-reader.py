#!/usr/bin/env python3
import logging
import sys
from numbers import Integral
from pathlib import Path

logger = logging.getLogger("reader")

# Necessary to load the local gguf package
sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf.gguf_reader import GGUFReader


def _decode_string_like_value(value):
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).split(b"\x00", 1)[0].decode("utf-8")
        except UnicodeDecodeError:
            return None

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, list) and value and all(isinstance(v, Integral) and 0 <= int(v) <= 255 for v in value):
        try:
            raw = bytes(int(v) for v in value)
            return raw.split(b"\x00", 1)[0].decode("utf-8")
        except UnicodeDecodeError:
            return None

    return None


def _format_field_value(field):
    value = field.parts[field.data[0]]
    decoded = _decode_string_like_value(value)
    if decoded is not None:
        return decoded
    return value


def read_gguf_file(gguf_file_path):
    """
    Reads and prints key-value pairs and tensor information from a GGUF file in an improved format.

    Parameters:
    - gguf_file_path: Path to the GGUF file.
    """

    reader = GGUFReader(gguf_file_path)

    # List all key-value pairs in a columnized format
    print("Key-Value Pairs:") # noqa: NP100
    max_key_length = max(len(key) for key in reader.fields.keys())
    for key, field in reader.fields.items():
        value = _format_field_value(field)
        print(f"{key:{max_key_length}} : {value}") # noqa: NP100
    print("----") # noqa: NP100

    # List all tensors
    print("Tensors:") # noqa: NP100
    tensor_info_format = "{:<30} | Shape: {:<15} | Size: {:<12} | Quantization: {}"
    print(tensor_info_format.format("Tensor Name", "Shape", "Size", "Quantization")) # noqa: NP100
    print("-" * 80) # noqa: NP100
    for tensor in reader.tensors:
        shape_str = "x".join(map(str, tensor.shape))
        size_str = str(tensor.n_elements)
        quantization_str = tensor.tensor_type.name
        print(tensor_info_format.format(tensor.name, shape_str, size_str, quantization_str)) # noqa: NP100


if __name__ == '__main__':
    if len(sys.argv) < 2:
        logger.info("Usage: gguf-reader.py <path_to_gguf_file>")
        sys.exit(1)
    gguf_file_path = sys.argv[1]
    read_gguf_file(gguf_file_path)