from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from converter.python.onnx2gguf import (
    _load_onnx_model_for_ggml,
    _map_axis_rank5,
    _map_perm_rank5,
    export_graph_and_weights,
    export_ir_json,
    parse_onnx_to_ir,
    resolve_output_paths,
)


class OnnxToIrTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model_path = Path("assets/models/yolo/raw/blank_16.onnx")

    def test_parse_onnx_to_ir_returns_graph(self) -> None:
        model_ir = parse_onnx_to_ir(self.model_path)
        self.assertGreaterEqual(model_ir.ir_version, 1)
        self.assertIsNotNone(model_ir.graph)
        self.assertIsInstance(model_ir.graph.nodes, list)
        self.assertIsInstance(model_ir.graph.inputs, list)
        self.assertIsInstance(model_ir.graph.outputs, list)
        self.assertTrue(all(t.values is None for t in model_ir.graph.initializers))

    def test_parse_onnx_to_ir_can_include_initializer_values(self) -> None:
        model_ir = parse_onnx_to_ir(self.model_path, include_initializer_values=True)
        if model_ir.graph.initializers:
            self.assertTrue(any(t.values is not None for t in model_ir.graph.initializers))

    def test_export_ir_json_generates_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "graph.json"
            exported = export_ir_json(self.model_path, output_path)
            self.assertTrue(exported.exists())
            payload = json.loads(exported.read_text(encoding="utf-8"))
            self.assertIn("graph", payload)
            self.assertIn("nodes", payload["graph"])

    def test_export_graph_and_weights_generates_both_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            graph_output = Path(temp_dir) / "graph.json"
            weights_output = Path(temp_dir) / "weights.gguf"
            graph_path, weights_path = export_graph_and_weights(
                model_path=self.model_path,
                graph_output_path=graph_output,
                weights_output_path=weights_output,
            )
            self.assertTrue(graph_path.exists())
            self.assertTrue(weights_path.exists())
            self.assertGreater(weights_path.stat().st_size, 0)

    def test_constant_tensor_values_not_embedded_in_graph_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "graph.json"
            exported = export_ir_json(self.model_path, output_path)
            payload = json.loads(exported.read_text(encoding="utf-8"))
            constant_nodes = [n for n in payload["graph"]["nodes"] if n["op_type"] == "Constant"]
            self.assertGreater(len(constant_nodes), 0)
            for node in constant_nodes:
                value_attr = node["attributes"].get("value")
                if isinstance(value_attr, dict):
                    self.assertNotIn("values", value_attr)

    def test_resolve_output_paths_uses_input_stem(self) -> None:
        graph_path, weights_path = resolve_output_paths(self.model_path, "assets/models/yolo/converted")
        self.assertEqual(graph_path.name, "blank_16_graph.json")
        self.assertEqual(weights_path.name, "blank_16_weights.gguf")

    def test_resolve_output_paths_defaults_to_input_directory(self) -> None:
        graph_path, weights_path = resolve_output_paths(self.model_path)
        self.assertEqual(graph_path.parent, self.model_path.parent)
        self.assertEqual(weights_path.parent, self.model_path.parent)

    def test_map_rank5_axis_and_perm(self) -> None:
        self.assertEqual(_map_axis_rank5(4), 3)
        self.assertEqual(_map_axis_rank5(-1), 3)
        self.assertEqual(_map_axis_rank5(3), 2)
        self.assertEqual(
            _map_perm_rank5([0, 1, 3, 4, 2], pair_in=(3, 4), pair_out=(2, 3)),
            [0, 1, 3, 2],
        )

    def test_map_rank5_perm_rejects_invalid_hw_order(self) -> None:
        with self.assertRaises(ValueError):
            _map_perm_rank5([0, 3, 2, 1, 4], pair_in=(2, 3), pair_out=(2, 3))

    def test_rank_six_tensor_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_info = helper.make_tensor_value_info(
                "x",
                TensorProto.FLOAT,
                [1, 2, 3, 4, 5, 6],
            )
            output_info = helper.make_tensor_value_info(
                "y",
                TensorProto.FLOAT,
                [1, 2, 3, 4, 5, 6],
            )
            node = helper.make_node("Identity", inputs=["x"], outputs=["y"])
            graph = helper.make_graph([node], "rank6_graph", [input_info], [output_info])
            model = helper.make_model(graph)
            model_path = Path(temp_dir) / "rank6.onnx"
            onnx.save(model, str(model_path))

            with self.assertRaises(ValueError):
                _load_onnx_model_for_ggml(model_path)


if __name__ == "__main__":
    unittest.main()
