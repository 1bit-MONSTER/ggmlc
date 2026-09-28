"""Real checkpoint loaders for OpenCLIP models"""

from __future__ import annotations

import torch
from ggmlc.pipeline import VisionPreprocessor
from open_clip import create_model_from_pretrained
from open_clip.transform import MaybeConvertMode, MaybeToTensor
from timm.utils import reparameterize_model
from torchvision.transforms.transforms import CenterCrop, Normalize, Resize

__all__ = [
    "load_openclip_image_encoder_model",
]

# preprocess
# Compose(
#     Resize(size=256, interpolation=bicubic, max_size=None, antialias=True)
#     CenterCrop(size=(256, 256))
#     MaybeConvertMode()
#     MaybeToTensor()
#     Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
#               std=(0.26862954, 0.26130258, 0.27577711))
# )

def make_vision_preprocessor(preprocess) -> VisionPreprocessor:
    """Creates a GGMLC VisionPreprocessor."""
    kwargs = {}
    for transform in preprocess.transforms:
        if isinstance(transform, Resize):
            target_size = (transform.size, transform.size) \
                if isinstance(transform.size, int) else transform.size
            kwargs["target_size"] = target_size
        elif isinstance(transform, Normalize):
            kwargs["mean"] = transform.mean
            kwargs["std"] = transform.std
        elif isinstance(transform, (CenterCrop, MaybeConvertMode, MaybeToTensor)):
            continue  # Ignore
        else:
            raise TypeError(f"Unsupported transform: {transform}")
    return VisionPreprocessor(**kwargs)


def load_openclip_image_encoder_model(
    variant: str = "hf-hub:timm/MobileCLIP2-S2-OpenCLIP",
    with_preprocessor: bool = True,
    **kwargs
) -> tuple[torch.nn.Module, tuple[torch.Tensor, ...], list[str], VisionPreprocessor]:
    """Loads OpenCLIP pretrained image encoder model.
    """

    if "return_transform" in kwargs:
        raise ValueError("return_transform is not supported; use `with_preprocessor` instead")

    # Load the model and optionally preprocessing transforms
    create_result = create_model_from_pretrained(
        variant, return_transform=with_preprocessor, **kwargs)
    model = create_result if not with_preprocessor else create_result[0]

    # Prepare for inference/model exporting purposes
    model = reparameterize_model(model.visual.eval())

    # Create example input for the model
    image_size = model.image_size
    pixel_values = torch.randn(1, 3, image_size[0], image_size[1], dtype=torch.float32)
    example_input = (pixel_values,)
    input_names = ["pixel_values"]

    # Create a wrapper module for the image encoder
    class OpenCLIPImageEncoderWrapper(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, pixel_values):
            return torch.nn.functional.normalize(self.base(pixel_values), dim=-1)
    wrapped_model = OpenCLIPImageEncoderWrapper(model)

    if with_preprocessor:
        # Create a VisionPreprocessor for the model
        preprocess = create_result[1]
        vision_preprocessor = make_vision_preprocessor(preprocess)
        return wrapped_model, example_input, input_names, vision_preprocessor

    return wrapped_model, example_input, input_names
