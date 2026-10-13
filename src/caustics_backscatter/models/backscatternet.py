"""BackscatterNet: global backscatter parameters per image (Eq. 1 of the paper)."""

from __future__ import annotations

import math

import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.nn import functional as F

from caustics_backscatter.models.causticsnet import ImageNormalize
from caustics_backscatter.registry import MODELS


def _inverse_softplus(y: float) -> float:
    return y + math.log(-math.expm1(-y))


def backscatter(b_inf: torch.Tensor, beta: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """``B = B_inf * (1 - exp(-beta * Z))``.

    ``b_inf``/``beta``: ``[..., 3]``; ``depth``: ``[..., 1, H, W]`` -> ``[..., 3, H, W]``.
    """
    return b_inf[..., None, None] * (1.0 - torch.exp(-beta[..., None, None] * depth))


@MODELS.register("backscatter_global")
class BackscatterNet(nn.Module):
    """ResNeXt encoder + the paper's lightweight head (structure of the SC-SfMLearner pose
    decoder): conv-BN-ReLU layers on the deepest features followed by spatial mean-pooling
    into 6 numbers, ``B_inf = sigmoid(.)`` and ``beta = softplus(.)`` per RGB channel.

    ``forward(images [B, N, 3, H, W] in [0, 1])`` returns ``(b_inf, beta)``, each ``[B, N, 3]``.
    The last convolution starts at zero so initial outputs equal ``b_inf_init``/``beta_init``.
    """

    num_input_frames = 1

    def __init__(
        self,
        encoder_name: str = "resnext50_32x4d",
        encoder_weights: str | None = "imagenet",
        head_channels: int = 256,
        b_inf_init: float = 0.3,
        beta_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.normalize = ImageNormalize()
        self.encoder = smp.encoders.get_encoder(encoder_name, in_channels=3, depth=5, weights=encoder_weights)
        in_ch = self.encoder.out_channels[-1]
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_channels, kernel_size=1),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(head_channels, 6, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            self.out.bias[:3] = math.log(b_inf_init / (1.0 - b_inf_init))
            self.out.bias[3:] = _inverse_softplus(beta_init)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, num = images.shape[:2]
        features = self.encoder(self.normalize(images.flatten(0, 1)))[-1]
        params = self.out(self.head(features)).mean(dim=(2, 3)).float()
        params = params.unflatten(0, (bsz, num))
        return torch.sigmoid(params[..., :3]), F.softplus(params[..., 3:])
