import ggmlc
import jax
import jax.numpy as jnp
import numpy as np
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.frontend.jax import export_jax_fn
from ggmlc.ir.op import OpCode
from ggmlc.ir.tensor import StorageClass


def test_jax_frontend_export():
    def simple_fn(a, b):
        return jnp.sin(a) + jnp.cos(b)

    x = np.ones((4, 8), dtype=np.float32)
    y = np.ones((4, 8), dtype=np.float32)

    model = export_jax_fn(simple_fn, (x, y), input_names=["a", "b"], model_name="trig_model")
    g = model.main_graph

    assert len(g.inputs) == 2
    assert len(g.outputs) == 1
    assert g.get_tensor(g.inputs[0]).name == "a"
    assert g.get_tensor(g.inputs[1]).name == "b"
    assert g.get_tensor(g.outputs[0]).storage == StorageClass.OUTPUT
    g.validate_invariants()


def test_jax_frontend_params_and_inlining():
    def mlp(x, w, b):
        return jax.nn.relu(jnp.dot(x, w) + b)

    x = np.random.randn(2, 16).astype(np.float32)
    w = np.random.randn(16, 32).astype(np.float32)
    b = np.random.randn(32).astype(np.float32)

    model = export_jax_fn(
        mlp,
        (x, w, b),
        input_names=["x", "w", "b"],
        params={"w": w, "b": b},
        model_name="mlp_model",
    )
    g = model.main_graph

    assert len(g.inputs) == 1
    params = [tid for tid in g.parameters if g.get_tensor(tid).storage == StorageClass.PARAMETER]
    assert len(params) == 2
    assert g.get_tensor(g.inputs[0]).name == "x"
    g.validate_invariants()


def test_jax_multi_axis_sum_chains():
    """Multi-axis reduce_sum chains single-axis SUMs (ggml reduces one axis)."""

    def pool(x):
        return jnp.sum(x, axis=(1, 2))

    x = np.ones((2, 8, 8, 4), dtype=np.float32)
    model = export_jax_fn(pool, (x,), input_names=["x"], model_name="pool")
    g = model.main_graph
    sums = [n for n in g.nodes if n.opcode == OpCode.SUM]
    assert len(sums) == 2, [n.name for n in sums]
    for node in sums:
        assert node.attributes["dim"] in (1, 2), node.attributes
    ggml_graph = lower_to_ggml(g, enable_fusion=False)
    assert len(ggml_graph.nodes) >= len(sums)
    g.validate_invariants()


def test_jax_asymmetric_conv_padding():
    """TF 'SAME' convs with odd total padding keep exact shapes and values."""

    rng = np.random.default_rng(0)
    w = rng.standard_normal((5, 5, 3, 8)).astype(np.float32)
    dn = ("NHWC", "HWIO", "NHWC")

    def conv(x):
        return jax.lax.conv_general_dilated(
            x, w, (2, 2), "SAME", feature_group_count=1, dimension_numbers=dn
        )

    x = rng.standard_normal((1, 56, 56, 3)).astype(np.float32)
    model = export_jax_fn(conv, (x,), input_names=["x"], model_name="aconv")
    g = model.main_graph
    convs = [n for n in g.nodes if n.opcode == OpCode.CONV2D]
    assert convs, "expected a conv node"
    pads = [n for n in g.nodes if n.opcode == OpCode.CONCAT and "pad" in n.name]
    assert pads, "expected explicit zero-pad concats for the asymmetric sides"

    gguf_bytes = ggmlc.compile(model=conv, sample_inputs=(x,), model_name="aconv")
    runner = ggmlc.load(gguf_bytes, n_threads=2)
    out = np.asarray(runner(x))
    ref = np.asarray(conv(jax.numpy.asarray(x)))
    assert out.reshape(ref.shape).shape == ref.shape
    np.testing.assert_allclose(out.reshape(ref.shape), ref, atol=1e-5, rtol=1e-4)
