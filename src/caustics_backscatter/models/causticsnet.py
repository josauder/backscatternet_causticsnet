"""Dense per-pixel predictors (CausticsNet)."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import nn

from caustics_backscatter.registry import MODELS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNormalize(nn.Module):
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


@MODELS.register("smp_dense")
class SMPDensePredictor(nn.Module):
    """Any segmentation_models_pytorch architecture applied to each view independently.

    Multi-view interface: ``forward(images [B, N, 3, H, W] in [0, 1]) -> [B, N, C, H, W]``
    (one prediction per input view). Models that fuse several views implement the same
    signature with ``num_input_frames > 1``.
    """

    num_input_frames = 1

    def __init__(
        self,
        arch: str = "DeepLabV3Plus",
        encoder_name: str = "resnext50_32x4d",
        encoder_weights: str | None = "imagenet",
        out_channels: int = 3,
        **arch_kwargs,
    ) -> None:
        super().__init__()
        self.normalize = ImageNormalize()
        self.net = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=out_channels,
            activation=None,
            **arch_kwargs,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        bsz, num = images.shape[:2]
        out = self.net(self.normalize(images.flatten(0, 1)))
        return out.unflatten(0, (bsz, num))
