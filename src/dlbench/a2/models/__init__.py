"""Common logits contract for both A2 models."""
from .unet import UNet


def count_parameters(model):
    """Count unique parameter elements by requires_grad; exclude buffers."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    return {"total_params": total, "trainable_params": trainable,
            "frozen_params": total - trainable}


def build_model(config, *, pretrained=True):
    if config["name"] == "unet":
        return UNet(config["base_channels"])
    if config["name"] == "segformer":
        from .segformer import SegFormer
        return SegFormer(config["checkpoint"], config["revision"], pretrained=pretrained)
    raise ValueError(f"Unknown A2 model: {config['name']}")
