import numpy as np
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode
from ggmlc.ir.shape import Shape
from ggmlc.ir.tensor import DType, StorageClass
from ggmlc.serialization.gguf import serialize_to_gguf
from ggmlc.validation.numerical import run_compiled_model_wsl


def test_gguf_serialization_roundtrip_wsl():
    """Verify that a graph serialized to GGUF format can be loaded and executed by C++ runtime."""
    g = Graph("test_gguf_mlp")
    x = g.add_tensor("x", Shape([1, 4]), DType.F32, StorageClass.INPUT)
    w = g.add_tensor("w", Shape([4, 4]), DType.F32, StorageClass.PARAMETER)
    b = g.add_tensor("b", Shape([4]), DType.F32, StorageClass.PARAMETER)
    out = g.add_tensor("out", Shape([1, 4]), DType.F32, StorageClass.ACTIVATION)

    w.data = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    b.data = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    g.add_node(OpCode.LINEAR, inputs=[x.id, w.id, b.id], outputs=[out.id])
    g.inputs = [x.id]
    g.outputs = [out.id]
    g.parameters = [w.id, b.id]

    ggml_graph = lower_to_ggml(g)
    gguf_bytes = serialize_to_gguf(ggml_graph)

    assert gguf_bytes[:4] == b"GGUF"

    x_val = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    inputs = {"x": x_val}

    res = run_compiled_model_wsl(gguf_bytes, inputs=inputs, output_tensor_ids=[out.id])
    out_val = res[out.id]

    expected = (x_val @ w.data.T + b.data).reshape(out_val.shape)
    np.testing.assert_allclose(out_val, expected, rtol=1e-5, atol=1e-5)


def test_gguf_long_tensor_names_load_natively():
    """Parameter names >= GGML_MAX_NAME (64) must be shortened consistently.

    The native loader binds weight data by name, so the tensor-info table
    and the graph-spec JSON must share the shortened names.
    """
    from ggmlc import _runtime
    from ggmlc.serialization.gguf import deserialize_ggml_graph

    g = Graph("long_names")
    x = g.add_tensor("x", Shape([1, 4]), DType.F32, StorageClass.INPUT)
    long_name = "visual.trunk.stages.3.blocks.11.mlp.fc1.weight.with.extra.suffix"
    assert len(long_name) >= 64
    w = g.add_tensor(long_name, Shape([4, 4]), DType.F32, StorageClass.PARAMETER)
    out = g.add_tensor("out", Shape([1, 4]), DType.F32, StorageClass.ACTIVATION)
    w.data = np.eye(4, dtype=np.float32)
    g.add_node(OpCode.LINEAR, inputs=[x.id, w.id], outputs=[out.id])
    g.inputs = [x.id]
    g.outputs = [out.id]
    g.parameters = [w.id]

    ggml_graph = lower_to_ggml(g)
    gguf_bytes = bytes(serialize_to_gguf(ggml_graph))
    assert gguf_bytes[:4] == b"GGUF"
    _runtime.ModelLoader.load_from_bytes(gguf_bytes)  # native parser accepts
    g2 = deserialize_ggml_graph(gguf_bytes)  # python reader agrees
    assert len(g2.tensors) == len(ggml_graph.tensors)
