"""End-to-end tests for OpenCLIP models."""

import torch
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.runtime.runner import ModelRunner
from ggmlc.serialization.graph import serialize_ggml_graph
from ggmlc.validation.numerical import check_numerical_accuracy

from examples.models.openclip_model import load_openclip_image_encoder_model


def test_mobileclip2_s2_cpu():
    """Verifies MobileCLIP2-S2 on CPU backend."""

    model_name = "MobileCLIP2-S2"
    variant = "hf-hub:timm/MobileCLIP2-S2-OpenCLIP"

    [model, example_input, input_names] = load_openclip_image_encoder_model(
        variant=variant, with_preprocessor=False
    )

    assert input_names == ["pixel_values"]

    with torch.inference_mode():
        ref = model(*example_input).numpy()

    exported = export_torch_model(model, example_input, model_name=model_name)
    assert len(exported.main_graph.nodes) > 0

    ggml_graph = lower_to_ggml(exported.main_graph)
    ser_bytes = serialize_ggml_graph(ggml_graph)

    runner = ModelRunner(ser_bytes, device="cpu")
    act = runner(*[x.numpy() for x in example_input])
    if isinstance(act, dict):
        act = next(iter(act.values()))
    act = act.reshape(ref.shape)

    result = check_numerical_accuracy(ref, act, atol=1e-3)
    assert result.passed, f"{model_name} numerical parity failed: {result}"


if __name__ == "__main__":
    test_mobileclip2_s2_cpu()
