"""Photometric loss of Eq. 4 (same SSIM and weighting as deepreefmap's SC-SfMLearner)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SSIM(nn.Module):
    """Per-pixel ``(1 - SSIM) / 2`` with 3x3 windows and reflection padding."""

    c1 = 0.01**2
    c2 = 0.03**2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        y = F.pad(y, (1, 1, 1, 1), mode="reflect")
        mu_x = F.avg_pool2d(x, 3, 1)
        mu_y = F.avg_pool2d(y, 3, 1)
        sigma_x = F.avg_pool2d(x * x, 3, 1) - mu_x**2
        sigma_y = F.avg_pool2d(y * y, 3, 1) - mu_y**2
        sigma_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y
        n = (2 * mu_x * mu_y + self.c1) * (2 * sigma_xy + self.c2)
        d = (mu_x**2 + mu_y**2 + self.c1) * (sigma_x + sigma_y + self.c2)
        return torch.clamp((1 - n / d) * 0.5, 0, 1)


_ssim = SSIM()


def photometric_map(a: torch.Tensor, b: torch.Tensor, ssim_weight: float = 0.85) -> torch.Tensor:
    """``(1 - w) * |a - b| + w * (1 - SSIM(a, b)) / 2`` per pixel and channel, ``[N, C, H, W]``."""
    a = a.float()
    b = b.float()
    diff = (a - b).abs()
    if ssim_weight > 0:
        diff = (1.0 - ssim_weight) * diff + ssim_weight * _ssim(a, b)
    return diff


def masked_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted mean over all elements; ``weight [N, 1, H, W]`` broadcasts over channels."""
    weight = weight.expand_as(values)
    return (values * weight).sum() / weight.sum().clamp(min=1.0)


def stationary_automask(target: torch.Tensor, warped: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """1 where warped L1 is strictly below unwarped L1 (SC-SfMLearner ``with_auto_mask``).

    The comparison is the channel-mean absolute error, and the warped error is clamped
    to ``[0, 1]`` first, matching ``compute_pairwise_loss``. Ties are dropped, so an
    identity warp contributes nothing. Returns ``[N, 1, H, W]``.
    """
    warped_l1 = (target - warped).abs().clamp(0, 1).mean(dim=1, keepdim=True)
    identity_l1 = (target - source).abs().mean(dim=1, keepdim=True)
    return (warped_l1 < identity_l1).to(dtype=target.dtype)
