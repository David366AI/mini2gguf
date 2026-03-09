from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TensorIR:
    name: str
    data_type: str
    dims: list[int | str]
    values: list[Any] | None = None


@dataclass(slots=True)
class ValueInfoIR:
    name: str
    data_type: str
    dims: list[int | str]


@dataclass(slots=True)
class NodeIR:
    name: str
    op_type: str
    domain: str
    inputs: list[str]
    outputs: list[str]
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphIR:
    name: str
    inputs: list[ValueInfoIR]
    outputs: list[ValueInfoIR]
    value_info: list[ValueInfoIR]
    initializers: list[TensorIR]
    nodes: list[NodeIR]


@dataclass(slots=True)
class ModelIR:
    ir_version: int
    producer_name: str
    producer_version: str
    domain: str
    model_version: int
    opset_imports: dict[str, int]
    graph: GraphIR
    model_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
