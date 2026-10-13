from __future__ import annotations

import torch


def tv_squared(x: torch.Tensor) -> torch.Tensor:
    """Total variation with squared finite differences: ``mean(dx^2) + mean(dy^2)``."""
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return dx.pow(2).mean() + dy.pow(2).mean()


def l2(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean()


def clip_loss(image: torch.Tensor) -> torch.Tensor:
    """Eq. 8: ``||min(max(I, 0), 1) - I||^2`` (mean over pixels)."""
    return (image.clamp(0.0, 1.0) - image).pow(2).mean()
