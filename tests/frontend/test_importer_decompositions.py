"""Structure-level tests for importer decompositions and lowering guards.

Fast (no execution, no compilation): assert the graph shapes the
importer/lowering produce for patterns the branch fixes.
"""

import pytest
import torch
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.dialect.ggml.ops import GGMLOpCode
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.ir.dtype import DType
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode
from ggmlc.ir.shape import Shape
from ggmlc.ir.tensor import StorageClass
from torch import nn


class QKVSplit(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(32, 3 * 4 * 8, bias=False)

    def forward(self, x):
        b, _c, h, w = x.shape
        qkv = self.proj(x.flatten(2).transpose(1, 2))
        qkv = qkv.reshape(b, h * w, 3, 4, 8).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        return q + 2 * k + 3 * v


def _exported(model, args):
    return export_torch_model(model, args, model_name="t", enable_fusion=False)


def test_5d_permute_decomposes_to_4d():
    """Non-batch-first 5D permutes become squeeze + 4D permute (no 5D survives)."""
    torch.manual_seed(0)
    exported = _exported(QKVSplit().eval(), (torch.randn(1, 32, 4, 4),))
    permutes = [n for n in exported.main_graph.nodes if n.opcode == OpCode.PERMUTE]
    assert permutes, "expected a permute node"
    for node in permutes:
        in_t = exported.main_graph.get_tensor(node.inputs[0])
        assert len(in_t.shape.dims) <= 4, (node.name, in_t.shape.dims)
        assert len(node.attributes["dims"]) <= 4, (node.name, node.attributes)
    sq = [n for n in exported.main_graph.nodes if n.name.endswith("_sq5")]
    assert sq, "expected the squeeze helper node"


def test_5d_permute_lowes_to_ggml_permute():
    """The decomposed permute lowers to PERMUTE with concrete 4D axes."""
    torch.manual_seed(0)
    exported = _exported(QKVSplit().eval(), (torch.randn(1, 32, 4, 4),))
    ggml_graph = lower_to_ggml(exported.main_graph, enable_fusion=False)
    perms = [op for op in ggml_graph.nodes if op.opcode == GGMLOpCode.GGML_OP_PERMUTE]
    assert perms, [op.opcode for op in ggml_graph.nodes]
    for op in perms:
        axes = [op.attributes[f"axis{i}"] for i in range(4)]
        assert all(0 <= a <= 3 for a in axes), (op.name, axes)
        assert sorted(axes) == [0, 1, 2, 3], (op.name, axes)


def _permute5_graph():
    g = Graph(name="perm5")
    x = g.add_tensor("x", Shape.from_tuple((2, 3, 4, 5, 6)), DType.F32, StorageClass.INPUT)
    o = g.add_tensor("y", Shape.from_tuple((4, 2, 5, 3, 6)), DType.F32, StorageClass.OUTPUT)
    g.inputs = [x.id]
    g.outputs = [o.id]
    g.add_op(OpCode.PERMUTE, [x.id], [o.id], attributes={"dims": [2, 0, 3, 1, 4]}, name="perm")
    return g


def test_lowering_rejects_inexpressible_5d_permute():
    """A 5D permute with no unit dims to squeeze must fail loudly, not scramble."""
    with pytest.raises(NotImplementedError):
        lower_to_ggml(_permute5_graph(), enable_fusion=False)


def test_mean_chain_structure():
    """Multi-dim means chain single-dim reductions over keepdim intermediates."""

    class MeanTwo(nn.Module):
        def forward(self, x):
            return x.mean(dim=(1, 3), keepdim=True)

    exported = _exported(MeanTwo().eval(), (torch.randn(2, 4, 8, 8),))
    means = [n for n in exported.main_graph.nodes if n.opcode == OpCode.MEAN]
    assert len(means) == 2, [n.name for n in means]
    for node in means:
        assert isinstance(node.attributes["dim"], int), node.attributes
    ggml_graph = lower_to_ggml(exported.main_graph, enable_fusion=False)
    gmeans = [op for op in ggml_graph.nodes if op.opcode == GGMLOpCode.GGML_OP_MEAN]
    assert len(gmeans) == 2, [op.name for op in ggml_graph.nodes]
    assert {op.attributes["ggml_dim"] for op in gmeans} == {2, 0}, [
        (op.name, op.attributes) for op in gmeans
    ]


def test_gelu_approximate_mapping():
    """exact (default) GELU lowers to ERF(16); tanh lowers to GELU(8)."""

    class GeluBoth(nn.Module):
        def forward(self, x):
            a = torch.nn.functional.gelu(x)
            b = torch.nn.functional.gelu(x, approximate="tanh")
            return a + b

    exported = _exported(GeluBoth().eval(), (torch.randn(2, 8),))
    gelus = {n.name: n for n in exported.main_graph.nodes if n.opcode == OpCode.GELU}
    assert len(gelus) == 2, [n.opcode for n in exported.main_graph.nodes]
    approx = sorted(n.attributes["approximate"] for n in gelus.values())
    assert approx == ["none", "tanh"], approx
    ggml_graph = lower_to_ggml(exported.main_graph, enable_fusion=False)
    codes = sorted(
        op.attributes["unary_op"]
        for op in ggml_graph.nodes
        if op.opcode == GGMLOpCode.GGML_OP_UNARY
    )
    assert codes == [8, 16], codes


def test_expand_as_mapping():
    """aten.expand_as maps to EXPAND with the other tensor's shape."""

    class ExpandAs(nn.Module):
        def forward(self, x, y):
            return x.expand_as(y) + y

    exported = _exported(ExpandAs().eval(), (torch.randn(1, 4, 1, 1), torch.randn(1, 4, 8, 8)))
    expands = [n for n in exported.main_graph.nodes if n.opcode == OpCode.EXPAND]
    assert len(expands) == 1, [n.opcode for n in exported.main_graph.nodes]
    out_t = exported.main_graph.get_tensor(expands[0].outputs[0])
    assert [d.evaluate({}) for d in out_t.shape.dims] == [1, 4, 8, 8]


def test_lowering_rejects_multi_axis_sum():
    """A multi-axis SUM must be chained by the frontend, never silently sliced."""
    from ggmlc.ir.dtype import DType
    from ggmlc.ir.graph import Graph
    from ggmlc.ir.op import OpCode
    from ggmlc.ir.shape import Shape
    from ggmlc.ir.tensor import StorageClass

    g = Graph(name="multisum")
    x = g.add_tensor("x", Shape.from_tuple((1, 4, 8, 8)), DType.F32, StorageClass.INPUT)
    o = g.add_tensor("y", Shape.from_tuple((1, 1, 1, 8)), DType.F32, StorageClass.OUTPUT)
    g.inputs = [x.id]
    g.outputs = [o.id]
    g.add_op(OpCode.SUM, [x.id], [o.id], attributes={"axes": (1, 2)}, name="bad_sum")
    with pytest.raises(NotImplementedError):
        lower_to_ggml(g, enable_fusion=False)


def test_torch_multi_dim_sum_chains():
    """Multi-dim torch sums chain single-axis SUMs instead of dropping axes."""

    class SumTwo(nn.Module):
        def forward(self, x):
            return x.sum(dim=(1, 2), keepdim=True)

    exported = _exported(SumTwo().eval(), (torch.randn(2, 4, 8, 8),))
    sums = [n for n in exported.main_graph.nodes if n.opcode == OpCode.SUM]
    assert len(sums) == 2, [n.name for n in sums]
    ggml_graph = lower_to_ggml(exported.main_graph, enable_fusion=False)
    gsums = [op for op in ggml_graph.nodes if op.opcode == GGMLOpCode.GGML_OP_SUM_ROWS]
    assert len(gsums) == 2, [op.name for op in ggml_graph.nodes]
