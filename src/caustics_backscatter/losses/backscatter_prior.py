"""Eq. 11: far-away pixels should have backscatter equal to the mean far-away image color."""

from __future__ import annotations

import torch


def far_backscatter_loss(
    backscatter: torch.Tensor,
    image: torch.Tensor,
    depth: torch.Tensor,
    quantile: float = 0.95,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """``L_B = Z * ||B - mean_far(I)||^2`` averaged over pixels beyond the depth quantile.

    Args:
        backscatter, image: ``[N, 3, H, W]``.
        depth: ``[N, 1, H, W]``.
        quantile: per-image depth quantile defining the far region (0.95 = furthest 5%).
        weight: optional ``[N, 1, H, W]`` pixel weights (e.g. masking moving objects).
    """
    flat = depth.flatten(1).float()
    threshold = torch.quantile(flat, quantile, dim=1).view(-1, 1, 1, 1)
    far = (depth > threshold).float()
    if weight is not None:
        far = far * weight
    count = far.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
    mean_color = ((image * far).sum(dim=(2, 3), keepdim=True) / count).detach()
    per_pixel = depth * (backscatter - mean_color).pow(2).sum(dim=1, keepdim=True)
    return (per_pixel * far).sum() / far.sum().clamp(min=1.0)
