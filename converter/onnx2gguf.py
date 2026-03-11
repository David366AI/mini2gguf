from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any

import gguf
import numpy as np
import onnx
from onnx import numpy_helper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from converter.internal.ir import GraphIR, ModelIR, NodeIR, TensorIR, ValueInfoIR


_ONNX_TYPE_NAME = {
    onnx.TensorProto.UNDEFINED: "UNDEFINED",
    onnx.TensorProto.FLOAT: "FLOAT",
    onnx.TensorProto.UINT8: "UINT8",
    onnx.TensorProto.INT8: "INT8",
    onnx.TensorProto.UINT16: "UINT16",
    onnx.TensorProto.INT16: "INT16",
    onnx.TensorProto.INT32: "INT32",
    onnx.TensorProto.INT64: "INT64",
    onnx.TensorProto.STRING: "STRING",
    onnx.TensorProto.BOOL: "BOOL",
    onnx.TensorProto.FLOAT16: "FLOAT16",
    onnx.TensorProto.DOUBLE: "DOUBLE",
    onnx.TensorProto.UINT32: "UINT32",
    onnx.TensorProto.UINT64: "UINT64",
    onnx.TensorProto.COMPLEX64: "COMPLEX64",
    onnx.TensorProto.COMPLEX128: "COMPLEX128",
    onnx.TensorProto.BFLOAT16: "BFLOAT16",
}


def _normalize_rank5_pair(pair: tuple[int, int]) -> tuple[int, int]:
    if len(pair) != 2:
        raise ValueError(f"Invalid rank-5 merge pair: {pair}")
    a, b = pair
    if a < 0 or b < 0 or a >= 5 or b >= 5 or b != a + 1:
        raise ValueError(f"Rank-5 merge pair must be adjacent and ordered, got: {pair}")
    return a, b


def _merge_dims_rank5_by_pair(dims: list[int | str], pair: tuple[int, int]) -> list[int | str]:
    if len(dims) != 5:
        return dims
    a, b = _normalize_rank5_pair(pair)
    h = dims[a]
    w = dims[b]
    if isinstance(h, int) and isinstance(w, int):
        merged_hw: int | str = h * w
    else:
        merged_hw = f"({h}*{w})"
    return [*dims[:a], merged_hw, *dims[b + 1 :]]


def _map_axis_rank5(axis: int, pair: tuple[int, int] = (2, 3)) -> int:
    a, b = _normalize_rank5_pair(pair)
    resolved = axis if axis >= 0 else axis + 5
    if resolved < 0 or resolved >= 5:
        raise ValueError(f"Invalid axis {axis} for rank-5 tensor")
    if resolved == b:
        return a
    if resolved > b:
        return resolved - 1
    return resolved


def _map_axes_rank5(axes: list[int], pair: tuple[int, int] = (2, 3)) -> list[int]:
    mapped: list[int] = []
    for axis in axes:
        new_axis = _map_axis_rank5(axis, pair=pair)
        if new_axis not in mapped:
            mapped.append(new_axis)
    return mapped


def _map_perm_rank5(
    perm: list[int],
    pair_in: tuple[int, int] = (2, 3),
    pair_out: tuple[int, int] = (2, 3),
) -> list[int]:
    if len(perm) != 5:
        return perm
    in_a, in_b = _normalize_rank5_pair(pair_in)
    out_a, out_b = _normalize_rank5_pair(pair_out)
    if [perm[out_a], perm[out_b]] != [in_a, in_b]:
        raise ValueError(f"Unsupported rank-5 transpose perm for merge pairs {pair_in}->{pair_out}: {perm}")

    out_axes = [0, 1, 2, 3, 4]
    out_axes.remove(out_b)
    mapped: list[int] = []
    for out_axis in out_axes:
        src_axis = in_a if out_axis == out_a else perm[out_axis]
        mapped.append(_map_axis_rank5(src_axis, pair=pair_in))
    return mapped


def _set_value_info_dims(value_info: onnx.ValueInfoProto, dims: list[int | str]) -> None:
    tensor_shape = value_info.type.tensor_type.shape
    tensor_shape.ClearField("dim")
    for dim_value in dims:
        dim = tensor_shape.dim.add()
        if isinstance(dim_value, int):
            dim.dim_value = dim_value
        else:
            dim.dim_param = str(dim_value)


def _reshape_rank5_array_to_rank4(array: np.ndarray, pair: tuple[int, int] = (2, 3)) -> np.ndarray:
    if array.ndim != 5:
        return array
    a, b = _normalize_rank5_pair(pair)
    shape = list(array.shape)
    merged = shape[a] * shape[b]
    new_shape = [*shape[:a], merged, *shape[b + 1 :]]
    return array.reshape(new_shape)


def _merge_reshape_shape_vector_if_rank5(array: np.ndarray, pair: tuple[int, int] = (2, 3)) -> np.ndarray:
    if array.ndim != 1 or array.size != 5:
        return array
    a, b = _normalize_rank5_pair(pair)
    values = array.tolist()
    h = int(values[a])
    w = int(values[b])
    merged_hw = -1 if h == -1 or w == -1 else h * w
    merged_values = [*values[:a], merged_hw, *values[b + 1 :]]
    merged = np.asarray(merged_values, dtype=array.dtype)
    return merged


def _tensor_rank_from_value_info(value_info: onnx.ValueInfoProto) -> int | None:
    if not value_info.type.HasField("tensor_type"):
        return None
    return len(value_info.type.tensor_type.shape.dim)


def _build_rank_map(model: onnx.ModelProto) -> dict[str, int]:
    rank_map: dict[str, int] = {}
    for value_info in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        rank = _tensor_rank_from_value_info(value_info)
        if rank is not None:
            rank_map[value_info.name] = rank
    for tensor in model.graph.initializer:
        rank_map[tensor.name] = len(tensor.dims)
    return rank_map


def _build_rank5_merge_pair_map(model: onnx.ModelProto, rank_map: dict[str, int]) -> dict[str, tuple[int, int]]:
    pair_map: dict[str, tuple[int, int]] = {
        tensor_name: (2, 3) for tensor_name, rank in rank_map.items() if rank == 5
    }

    updated = True
    while updated:
        updated = False
        for node in model.graph.node:
            if node.op_type != "Transpose" or not node.input or not node.output:
                continue
            input_name = node.input[0]
            output_name = node.output[0]
            if rank_map.get(input_name) != 5 or rank_map.get(output_name) != 5:
                continue

            perm_attr = next((attr for attr in node.attribute if attr.name == "perm"), None)
            if perm_attr is None or len(perm_attr.ints) != 5:
                continue
            perm = [int(v) for v in perm_attr.ints]

            out_pair = pair_map.get(output_name, (2, 3))
            candidate = (perm[out_pair[0]], perm[out_pair[1]])
            candidate = _normalize_rank5_pair(candidate)
            if pair_map.get(input_name) != candidate:
                pair_map[input_name] = candidate
                updated = True

    return pair_map


def _load_onnx_model_for_ggml_with_warnings(model_path: str | Path) -> tuple[onnx.ModelProto, list[str]]:
    model = onnx.load(str(model_path))
    warnings: list[str] = []
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass

    rank_map = _build_rank_map(model)
    merge_pair_map = _build_rank5_merge_pair_map(model, rank_map)

    for value_info in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        rank = _tensor_rank_from_value_info(value_info)
        if rank is None:
            continue
        if rank >= 6:
            raise ValueError(f"Tensor '{value_info.name}' has rank {rank}, which is not supported by ggml")
        if rank == 5:
            old_dims = _shape_to_dims(value_info.type.tensor_type.shape)
            pair = merge_pair_map.get(value_info.name, (2, 3))
            new_dims = _merge_dims_rank5_by_pair(old_dims, pair)
            _set_value_info_dims(value_info, new_dims)
            warnings.append(
                f"rank-5->4 merge for tensor '{value_info.name}': pair={pair}, dims {old_dims} -> {new_dims}"
            )

    reshape_shape_inputs: dict[str, tuple[int, int]] = {}
    for node in model.graph.node:
        if node.op_type == "Reshape" and len(node.input) >= 2:
            reshape_shape_inputs[node.input[1]] = merge_pair_map.get(node.output[0], (2, 3))

    for index, tensor in enumerate(model.graph.initializer):
        rank = len(tensor.dims)
        if rank >= 6:
            raise ValueError(f"Initializer '{tensor.name}' has rank {rank}, which is not supported by ggml")

        array = numpy_helper.to_array(tensor)
        rewritten = array
        if array.ndim == 5:
            pair = merge_pair_map.get(tensor.name, (2, 3))
            rewritten = _reshape_rank5_array_to_rank4(array, pair=pair)
            warnings.append(
                f"rank-5 initializer merge '{tensor.name}': pair={pair}, shape {list(array.shape)} -> {list(rewritten.shape)}"
            )
        if tensor.name in reshape_shape_inputs:
            old_vector = rewritten.tolist() if rewritten.ndim == 1 else None
            rewritten = _merge_reshape_shape_vector_if_rank5(rewritten, pair=reshape_shape_inputs[tensor.name])
            if old_vector is not None and rewritten.ndim == 1 and len(old_vector) == 5 and len(rewritten.tolist()) == 4:
                warnings.append(
                    f"reshape-shape initializer merge '{tensor.name}': pair={reshape_shape_inputs[tensor.name]}, "
                    f"shape {old_vector} -> {rewritten.tolist()}"
                )

        if rewritten is not array:
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(rewritten, name=tensor.name))

    for node in model.graph.node:
        node_rank = None
        for tensor_name in list(node.input) + list(node.output):
            if tensor_name in rank_map:
                node_rank = rank_map[tensor_name]
                break

        for attr in node.attribute:
            if attr.name == "perm" and attr.type == onnx.AttributeProto.INTS and len(attr.ints) == 5:
                input_name = node.input[0] if node.input else ""
                output_name = node.output[0] if node.output else ""
                pair_in = merge_pair_map.get(input_name, (2, 3))
                pair_out = merge_pair_map.get(output_name, (2, 3))
                old_perm = [int(v) for v in attr.ints]
                new_perm = _map_perm_rank5(old_perm, pair_in=pair_in, pair_out=pair_out)
                attr.ClearField("ints")
                attr.ints.extend(new_perm)
                warnings.append(
                    f"transpose perm rewrite at node '{node.name or node.op_type}': pair {pair_in}->{pair_out}, "
                    f"perm {old_perm} -> {new_perm}"
                )
                continue

            if attr.name == "axis" and attr.type == onnx.AttributeProto.INT and node_rank == 5:
                pair = None
                for tensor_name in list(node.input) + list(node.output):
                    if rank_map.get(tensor_name) == 5:
                        pair = merge_pair_map.get(tensor_name, (2, 3))
                        break
                old_axis = int(attr.i)
                new_axis = _map_axis_rank5(old_axis, pair=pair or (2, 3))
                attr.i = new_axis
                warnings.append(
                    f"axis rewrite at node '{node.name or node.op_type}': pair={pair or (2, 3)}, axis {old_axis} -> {new_axis}"
                )
                continue

            if attr.name == "axes" and attr.type == onnx.AttributeProto.INTS and node_rank == 5:
                pair = None
                for tensor_name in list(node.input) + list(node.output):
                    if rank_map.get(tensor_name) == 5:
                        pair = merge_pair_map.get(tensor_name, (2, 3))
                        break
                old_axes = [int(v) for v in attr.ints]
                new_axes = _map_axes_rank5(old_axes, pair=pair or (2, 3))
                attr.ClearField("ints")
                attr.ints.extend(new_axes)
                warnings.append(
                    f"axes rewrite at node '{node.name or node.op_type}': pair={pair or (2, 3)}, "
                    f"axes {old_axes} -> {new_axes}"
                )
                continue

            if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                tensor = attr.t
                if len(tensor.dims) >= 6:
                    raise ValueError(
                        f"Constant tensor in node '{node.name or node.op_type}' has rank {len(tensor.dims)}, unsupported"
                    )
                rewritten = numpy_helper.to_array(tensor)
                if rewritten.ndim == 5:
                    constant_pair = (2, 3)
                    if node.output:
                        constant_pair = merge_pair_map.get(node.output[0], (2, 3))
                    old_shape = list(rewritten.shape)
                    rewritten = _reshape_rank5_array_to_rank4(rewritten, pair=constant_pair)
                    warnings.append(
                        f"rank-5 constant merge at node '{node.name or node.op_type}': pair={constant_pair}, "
                        f"shape {old_shape} -> {list(rewritten.shape)}"
                    )
                if node.op_type == "Constant" and node.output and node.output[0] in reshape_shape_inputs:
                    old_vector = rewritten.tolist() if rewritten.ndim == 1 else None
                    rewritten = _merge_reshape_shape_vector_if_rank5(
                        rewritten,
                        pair=reshape_shape_inputs[node.output[0]],
                    )
                    if old_vector is not None and rewritten.ndim == 1 and len(old_vector) == 5 and len(rewritten.tolist()) == 4:
                        warnings.append(
                            f"reshape-shape constant merge at node '{node.name or node.op_type}': "
                            f"pair={reshape_shape_inputs[node.output[0]]}, shape {old_vector} -> {rewritten.tolist()}"
                        )
                if rewritten.shape != tuple(tensor.dims):
                    attr.t.CopyFrom(numpy_helper.from_array(rewritten, name=tensor.name))

    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass
    return model, warnings


def _load_onnx_model_for_ggml(model_path: str | Path) -> onnx.ModelProto:
    model, _ = _load_onnx_model_for_ggml_with_warnings(model_path)
    return model


def _tensor_type_name(elem_type: int) -> str:
    return _ONNX_TYPE_NAME.get(elem_type, f"TYPE_{elem_type}")


def _make_unique_tensor_name(base: str, used_names: set[str]) -> str:
    if base not in used_names:
        used_names.add(base)
        return base
    index = 1
    while True:
        candidate = f"{base}#{index}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        index += 1


def _collect_constant_tensor_map(model: onnx.ModelProto) -> tuple[dict[tuple[int, str], str], dict[str, np.ndarray]]:
    used_names = {initializer.name for initializer in model.graph.initializer}
    tensor_ref_map: dict[tuple[int, str], str] = {}
    tensor_data_map: dict[str, np.ndarray] = {}

    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.type != onnx.AttributeProto.TENSOR:
                continue
            base_name = attr.t.name or f"const::{node.name or f'node_{node_index}'}::{attr.name}"
            tensor_name = _make_unique_tensor_name(base_name, used_names)
            tensor_ref_map[(node_index, attr.name)] = tensor_name
            tensor_data_map[tensor_name] = np.asarray(numpy_helper.to_array(attr.t))

    return tensor_ref_map, tensor_data_map


def _shape_to_dims(tensor_shape: onnx.TensorShapeProto) -> list[int | str]:
    dims: list[int | str] = []
    for dim in tensor_shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def _value_info_to_ir(value_info: onnx.ValueInfoProto) -> ValueInfoIR:
    tensor_type = value_info.type.tensor_type
    return ValueInfoIR(
        name=value_info.name,
        data_type=_tensor_type_name(tensor_type.elem_type),
        dims=_shape_to_dims(tensor_type.shape),
    )


def _attribute_value(attr: onnx.AttributeProto, tensor_ref_name: str | None = None) -> Any:
    t = onnx.AttributeProto
    if attr.type == t.FLOAT:
        return float(attr.f)
    if attr.type == t.INT:
        return int(attr.i)
    if attr.type == t.STRING:
        return attr.s.decode("utf-8", errors="ignore")
    if attr.type == t.FLOATS:
        return [float(v) for v in attr.floats]
    if attr.type == t.INTS:
        return [int(v) for v in attr.ints]
    if attr.type == t.STRINGS:
        return [v.decode("utf-8", errors="ignore") for v in attr.strings]
    if attr.type == t.TENSOR:
        return {
            "name": tensor_ref_name or attr.t.name,
            "data_type": _tensor_type_name(attr.t.data_type),
            "dims": list(attr.t.dims),
        }
    if attr.type == t.GRAPH:
        return {"graph_name": attr.g.name}
    if attr.type == t.SPARSE_TENSOR:
        return {"sparse_tensor": True}
    if attr.type == t.TYPE_PROTO:
        return {"type_proto": True}
    if attr.type == t.TENSORS:
        return [{"name": ten.name, "dims": list(ten.dims)} for ten in attr.tensors]
    if attr.type == t.GRAPHS:
        return [{"graph_name": g.name} for g in attr.graphs]
    if attr.type == t.SPARSE_TENSORS:
        return [{"sparse_tensor": True} for _ in attr.sparse_tensors]
    if attr.type == t.TYPE_PROTOS:
        return [{"type_proto": True} for _ in attr.type_protos]
    return None


def _tensor_to_ir(tensor: onnx.TensorProto, include_values: bool = False) -> TensorIR:
    dims = list(tensor.dims)
    values: list[Any] | None = None
    if include_values:
        try:
            values = numpy_helper.to_array(tensor).tolist()
        except Exception:
            values = [base64.b64encode(tensor.raw_data).decode("ascii")]
    return TensorIR(
        name=tensor.name,
        data_type=_tensor_type_name(tensor.data_type),
        dims=dims,
        values=values,
    )


def _model_to_ir(model: onnx.ModelProto, include_initializer_values: bool = False) -> ModelIR:
    graph = model.graph
    metadata = {entry.key: entry.value for entry in model.metadata_props}

    constant_tensor_ref_map, _ = _collect_constant_tensor_map(model)

    nodes: list[NodeIR] = []
    for index, node in enumerate(graph.node):
        attributes: dict[str, Any] = {}
        for attr in node.attribute:
            tensor_ref_name = constant_tensor_ref_map.get((index, attr.name))
            attributes[attr.name] = _attribute_value(attr, tensor_ref_name=tensor_ref_name)
        nodes.append(
            NodeIR(
                name=node.name or f"{node.op_type}_{index}",
                op_type=node.op_type,
                domain=node.domain,
                inputs=list(node.input),
                outputs=list(node.output),
                attributes=attributes,
            )
        )

    graph_ir = GraphIR(
        name=graph.name,
        inputs=[_value_info_to_ir(v) for v in graph.input],
        outputs=[_value_info_to_ir(v) for v in graph.output],
        value_info=[_value_info_to_ir(v) for v in graph.value_info],
        initializers=[_tensor_to_ir(t, include_initializer_values) for t in graph.initializer],
        nodes=nodes,
    )

    return ModelIR(
        ir_version=int(model.ir_version),
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        domain=model.domain,
        model_version=int(model.model_version),
        opset_imports={op.domain: int(op.version) for op in model.opset_import},
        graph=graph_ir,
        model_metadata=metadata,
    )


def model_ir_to_json_text(model_ir: ModelIR, indent: int | None = 2) -> str:
    if indent is None:
        return json.dumps(model_ir.to_dict(), ensure_ascii=False, separators=(",", ":"))
    return json.dumps(model_ir.to_dict(), ensure_ascii=False, indent=indent)


def graph_json_text_from_model(
    model: onnx.ModelProto,
    include_initializer_values: bool = False,
    indent: int | None = 2,
) -> str:
    model_ir = _model_to_ir(model, include_initializer_values=include_initializer_values)
    return model_ir_to_json_text(model_ir, indent=indent)


def parse_onnx_to_ir(model_path: str | Path, include_initializer_values: bool = False) -> ModelIR:
    model = _load_onnx_model_for_ggml(model_path)
    return _model_to_ir(model, include_initializer_values=include_initializer_values)


def export_ir_json_from_model(
    model: onnx.ModelProto,
    output_path: str | Path,
    include_initializer_values: bool = False,
    indent: int = 2,
) -> Path:
    graph_json = graph_json_text_from_model(
        model,
        include_initializer_values=include_initializer_values,
        indent=indent,
    )
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(graph_json, encoding="utf-8")
    return output_file


def export_ir_json(
    model_path: str | Path,
    output_path: str | Path,
    include_initializer_values: bool = False,
    indent: int = 2,
) -> Path:
    model = _load_onnx_model_for_ggml(model_path)
    return export_ir_json_from_model(
        model=model,
        output_path=output_path,
        include_initializer_values=include_initializer_values,
        indent=indent,
    )


def export_weights_gguf(
    model_path: str | Path,
    weights_output_path: str | Path,
    arch: str = "mini2gguf",
    extra_metadata: dict[str, str] | None = None,
    cast_weight_to_f16: bool = True,
    embed_graph: bool = True,
    graph_json_indent: int | None = None,
) -> Path:
    model = _load_onnx_model_for_ggml(model_path)
    embedded_graph_json: str | None = None
    if embed_graph:
        embedded_graph_json = graph_json_text_from_model(
            model,
            include_initializer_values=False,
            indent=graph_json_indent,
        )
    return export_weights_gguf_from_model(
        model=model,
        weights_output_path=weights_output_path,
        arch=arch,
        extra_metadata=extra_metadata,
        cast_weight_to_f16=cast_weight_to_f16,
        embedded_graph_json=embedded_graph_json,
    )


def export_weights_gguf_from_model(
    model: onnx.ModelProto,
    weights_output_path: str | Path,
    arch: str = "mini2gguf",
    extra_metadata: dict[str, str] | None = None,
    cast_weight_to_f16: bool = True,
    embedded_graph_json: str | None = None,
) -> Path:
    output_file = Path(weights_output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    writer = gguf.GGUFWriter(str(output_file), arch)
    writer.add_name(arch)

    def _add_string_metadata(key: str, value: str) -> None:
        if hasattr(writer, "add_string"):
            writer.add_string(key, value)
        elif hasattr(writer, "add_str"):
            writer.add_str(key, value)
        elif hasattr(writer, "add_key_value"):
            writer.add_key_value(key, value)
        else:
            raise RuntimeError("GGUFWriter does not expose a string metadata writer API")

    merged_metadata: dict[str, str] = {}
    if extra_metadata:
        merged_metadata.update({str(key): str(value) for key, value in extra_metadata.items()})
    if embedded_graph_json is not None and "model.graph" not in merged_metadata:
        merged_metadata["model.graph"] = embedded_graph_json

    if merged_metadata:
        for key, value in merged_metadata.items():
            _add_string_metadata(str(key), str(value))

    _, constant_tensor_data_map = _collect_constant_tensor_map(model)

    def _maybe_cast_weight_to_f16(tensor_name: str, array: np.ndarray) -> np.ndarray:
        if not cast_weight_to_f16:
            return array
        if "weight" not in tensor_name.lower():
            return array
        if np.issubdtype(array.dtype, np.floating) and array.dtype != np.float16:
            return array.astype(np.float16)
        return array

    try:
        written_names: set[str] = set()
        for initializer in model.graph.initializer:
            tensor = numpy_helper.to_array(initializer)
            array = np.asarray(tensor)
            tensor_name = initializer.name
            array = _maybe_cast_weight_to_f16(tensor_name, array)
            if tensor_name in written_names:
                continue
            writer.add_tensor(
                tensor_name,
                array,
                raw_shape=list(initializer.dims) if initializer.dims else None,
            )
            written_names.add(tensor_name)

        for tensor_name, array in constant_tensor_data_map.items():
            if tensor_name in written_names:
                continue
            gguf_array = np.asarray(array)
            gguf_array = _maybe_cast_weight_to_f16(tensor_name, gguf_array)
            writer.add_tensor(
                tensor_name,
                gguf_array,
                raw_shape=list(gguf_array.shape),
            )
            written_names.add(tensor_name)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()

    return output_file


def export_graph_and_weights(
    model_path: str | Path,
    graph_output_path: str | Path,
    weights_output_path: str | Path,
    include_initializer_values: bool = False,
    indent: int = 2,
    cast_weight_to_f16: bool = True,
    split_graph: bool = False,
) -> tuple[Path | None, Path]:
    model = _load_onnx_model_for_ggml(model_path)
    graph_json = graph_json_text_from_model(
        model,
        include_initializer_values=include_initializer_values,
        indent=indent,
    )
    graph_path: Path | None = None
    if split_graph:
        graph_path = Path(graph_output_path)
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_text(graph_json, encoding="utf-8")

    weights_path = export_weights_gguf_from_model(
        model=model,
        weights_output_path=weights_output_path,
        cast_weight_to_f16=cast_weight_to_f16,
        embedded_graph_json=None if split_graph else graph_json,
    )
    return graph_path, weights_path


def resolve_output_paths(input_model_path: str | Path, output_dir: str | Path | None = None) -> tuple[Path, Path]:
    input_path = Path(input_model_path)
    base_dir = Path(output_dir) if output_dir is not None else input_path.parent
    stem = input_path.stem
    graph_path = base_dir / f"{stem}.json"
    weights_path = base_dir / f"{stem}.gguf"
    return graph_path, weights_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert ONNX model to mini2gguf GGUF weights (graph is embedded by default)")
    parser.add_argument("-i", "--input", required=True, help="Input ONNX model path")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory (default: input model directory)",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indent")
    parser.add_argument(
        "--weight-f16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cast floating-point weight tensors to float16 in GGUF (default: enabled).",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Also export standalone <stem>.json and do not embed graph JSON into GGUF metadata.",
    )
    parser.add_argument(
        "--model-family",
        default=None,
        help="Optional metadata field model.family, e.g. yolo/crnn",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    auto_graph_output, auto_weights_output = resolve_output_paths(args.input, args.output_dir)
    graph_output = auto_graph_output
    weights_output = auto_weights_output
    model, rewrite_warnings = _load_onnx_model_for_ggml_with_warnings(args.input)
    graph_json = graph_json_text_from_model(
        model=model,
        include_initializer_values=False,
        indent=args.indent,
    )
    graph_path: Path | None = None
    if args.split:
        graph_path = graph_output
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_text(graph_json, encoding="utf-8")
    extra_metadata: dict[str, str] | None = None
    if args.model_family:
        extra_metadata = {"model.family": str(args.model_family)}

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
    metadata_items = [f"model.graph={graph_meta_state}"]
    if args.model_family:
        metadata_items.insert(0, f"model.family={args.model_family}")
    print("GGUF metadata written: " + ", ".join(metadata_items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
