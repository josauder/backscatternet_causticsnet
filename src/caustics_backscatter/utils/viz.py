"""Tensor -> uint8 image helpers for W&B panels."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def to_uint8(image: torch.Tensor) -> np.ndarray:
    """``[3, H, W]`` float in [0, 1] -> ``[H, W, 3]`` uint8."""
    return (image.detach().float().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()


def signed_to_uint8(residual: torch.Tensor, gain: float) -> np.ndarray:
    """Signed residual/caustics ``[3, H, W]`` shown as ``0.5 + gain * r``."""
    return to_uint8(0.5 + gain * residual)


def depth_to_uint8(depth: torch.Tensor) -> np.ndarray:
    """``[1, H, W]`` depth -> turbo-colored inverse depth, normalized per image."""
    disp = 1.0 / depth.detach().float().clamp(min=1e-6)
    lo, hi = torch.quantile(disp.flatten(), torch.tensor([0.02, 0.98], device=disp.device))
    norm = ((disp - lo) / (hi - lo).clamp(min=1e-6)).clamp(0, 1)[0]
    gray = (norm * 255).byte().cpu().numpy()
    return cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def mask_to_uint8(mask: torch.Tensor) -> np.ndarray:
    """``[H, W]`` bool/float -> ``[H, W, 3]`` uint8."""
    m = (mask.detach().float().clamp(0, 1) * 255).byte().cpu().numpy()
    return np.repeat(m[..., None], 3, axis=2)
