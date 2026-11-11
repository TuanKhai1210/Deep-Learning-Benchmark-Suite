"""Four downsampling stages, lightweight U-Net initialized from scratch."""
import torch
from torch import nn
from torch.nn import functional as F


def block(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
        nn.GroupNorm(8, outputs), nn.ReLU(inplace=True),
        nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
        nn.GroupNorm(8, outputs), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()
        if base_channels < 8 or base_channels % 8:
            raise ValueError("base_channels must be a positive multiple of 8")
        channels = [base_channels * 2**i for i in range(5)]
        self.down = nn.ModuleList([block(3, channels[0])] +
                                  [block(channels[i - 1], channels[i]) for i in range(1, 5)])
        self.up = nn.ModuleList([block(channels[i] + channels[i - 1], channels[i - 1])
                                for i in range(4, 0, -1)])
        self.head = nn.Conv2d(channels[0], 17, 1)

    def forward(self, image):
        if min(image.shape[-2:]) < 16:
            raise ValueError("U-Net requires spatial dimensions >=16")
        skips = []
        value = image
        for index, layer in enumerate(self.down):
            value = layer(F.max_pool2d(value, 2) if index else value)
            skips.append(value)
        for layer, skip in zip(self.up, reversed(skips[:-1])):
            value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            value = layer(torch.cat([skip, value], dim=1))
        return self.head(value)
