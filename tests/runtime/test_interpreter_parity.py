"""Interpreter (nanobind runtime) vs torch parity for ops touched on this branch.

The standalone suite covers the generated C++ path; these cover the
interpreter path over the same lowered graphs, so the two runners must
agree with each other (both are diffed against torch here).
"""

import ggmlc
import numpy as np
import pytest
import torch
from torch import nn

torch.manual_seed(0)


def _run_interpreter(model, args, atol=1e-5, rtol=1e-4):
    model = model.eval()
    with torch.inference_mode():
        ref = model(*args).detach().cpu().numpy()
    gguf_bytes = ggmlc.compile(model=model, sample_inputs=args, model_name="parity")
    runner = ggmlc.load(gguf_bytes, n_threads=2)
    out = runner(*[np.ascontiguousarray(a.detach().cpu().numpy()) for a in args])
    out = np.asarray(out, dtype=np.float32)
    # The runner pads outputs to 4D; compare values with ref reshaped alike.
    assert out.size == ref.size, (out.shape, ref.shape)
    ref = ref.reshape(out.shape)
    diff = np.abs(out - ref)
    tol = atol + rtol * np.abs(ref)
    bad = diff > tol
    assert not bad.any(), f"maxdiff={diff.max()} ref0={ref.flat[0]} out0={out.flat[0]}"
    return out


@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_interpreter_mean_each_dim(dim):
    """MEAN over every torch dim of a 4D input (ggml dims 3/2 use the swap path)."""

    class MeanDim(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.d = d

        def forward(self, x):
            return x.mean(dim=self.d, keepdim=True)

    _run_interpreter(MeanDim(dim), (torch.randn(1, 4, 8, 8),))


def test_interpreter_mean_no_keepdim():
    class MeanNoKD(nn.Module):
        def forward(self, x):
            return x.mean(dim=1)

    _run_interpreter(MeanNoKD(), (torch.randn(2, 4, 8, 8),))


def test_interpreter_mean_two_dims():
    class MeanTwo(nn.Module):
        def forward(self, x):
            return x.mean(dim=(2, 3), keepdim=True)

    _run_interpreter(MeanTwo(), (torch.randn(1, 2, 4, 8),))


@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_interpreter_sum_each_dim(dim):
    class SumDim(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.d = d

        def forward(self, x):
            return x.sum(dim=self.d, keepdim=True)

    _run_interpreter(SumDim(dim), (torch.randn(1, 4, 8, 8),), atol=1e-4)


def test_interpreter_gelu_exact():
    class GeluExact(nn.Module):
        def forward(self, x):
            return torch.nn.functional.gelu(x * 2.0)

    _run_interpreter(GeluExact(), (torch.randn(2, 16),))


def test_interpreter_gelu_tanh():
    class GeluTanh(nn.Module):
        def forward(self, x):
            return torch.nn.functional.gelu(x * 2.0, approximate="tanh")

    # ggml's tanh GELU is a fast approximation of torch's tanh formula.
    _run_interpreter(GeluTanh(), (torch.randn(2, 16),), atol=5e-3, rtol=1e-3)


def test_interpreter_qkv_split_permute():
    """QKV reshape/5D-permute/unbind plumbing must preserve head/slot order."""

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

    _run_interpreter(QKVSplit(), (torch.randn(1, 32, 4, 4),))


def test_interpreter_clamp_min_open():
    class ClampMin(nn.Module):
        def forward(self, x):
            return (x * 10.0).clamp_min(0.5)

    _run_interpreter(ClampMin(), (torch.randn(4, 16),))


def test_interpreter_sdpa_default_scale():
    class TinyAttention(nn.Module):
        def forward(self, q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    args = tuple(torch.randn(1, 2, 8, 16) for _ in range(3))
    _run_interpreter(TinyAttention(), args, atol=1e-4)


def test_interpreter_conv_bias():
    class ConvBias(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)

        def forward(self, x):
            return self.conv(x)

    _run_interpreter(ConvBias(), (torch.randn(1, 3, 16, 16),))


def test_interpreter_fused_conv_relu():
    class ConvRelu(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)

        def forward(self, x):
            return torch.relu(self.conv(x))

    _run_interpreter(ConvRelu(), (torch.randn(1, 3, 16, 16),))


def test_interpreter_expand_as():
    class ExpandAs(nn.Module):
        def forward(self, x, y):
            return x.expand_as(y) + y

    _run_interpreter(ExpandAs(), (torch.randn(1, 4, 1, 1), torch.randn(1, 4, 8, 8)))


def test_interpreter_sum_two_dims():
    class SumTwo(nn.Module):
        def forward(self, x):
            return x.sum(dim=(1, 2), keepdim=True)

    _run_interpreter(SumTwo(), (torch.randn(1, 4, 8, 8),), atol=1e-4)


def test_interpreter_sum_two_dims_no_keepdim():
    class SumTwoNoKD(nn.Module):
        def forward(self, x):
            return x.sum(dim=(2, 3))

    _run_interpreter(SumTwoNoKD(), (torch.randn(1, 4, 8, 8),), atol=1e-4)


def test_interpreter_mean_two_dims_no_keepdim():
    class MeanTwoNoKD(nn.Module):
        def forward(self, x):
            return x.mean(dim=(1, 2))

    _run_interpreter(MeanTwoNoKD(), (torch.randn(2, 4, 8, 8),))


def test_interpreter_depthwise_conv():
    class Depthwise(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(8, 8, 3, padding=1, groups=8)

        def forward(self, x):
            return self.conv(x)

    _run_interpreter(Depthwise(), (torch.randn(1, 8, 16, 16),))
