"""Inverse warping with a linear (pinhole) camera model.

Matches deepreefmap's SC-SfMLearner warping: pixel coordinates are normalized with
``2 * u / (W - 1) - 1`` and, by default, sampled with ``align_corners=False``. The
reprojection grid is computed once and can be reused to warp any per-pixel tensor of the
source view (images, network outputs, loss weights, labels).
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _pixel_grid(height: int, width: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    return torch.stack([xs, ys, torch.ones_like(xs)], dim=0).reshape(1, 3, -1)


def reprojection_grid(
    tgt_depth: torch.Tensor,
    src_from_tgt: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Where each target pixel lands in the source view.

    Args:
        tgt_depth: ``[B, 1, H, W]`` depth of the target view.
        src_from_tgt: ``[B, 4, 4]`` transform ``T_{src<-tgt}``.
        intrinsics: ``[B, 3, 3]`` pinhole intrinsics shared by both views.

    Returns:
        ``grid`` ``[B, H, W, 2]`` in normalized coordinates for ``grid_sample``,
        ``valid`` ``[B, 1, H, W]`` (1 where the point is in front of the source camera
        and projects inside the source image), and ``computed_depth`` ``[B, 1, H, W]``,
        the Z of each target point in the source camera.
    """
    batch, _, height, width = tgt_depth.shape
    depth = tgt_depth.float()
    k = intrinsics.float()
    transform = src_from_tgt.float()
    pixels = _pixel_grid(height, width, depth.dtype, depth.device)
    cam = (torch.linalg.inv(k) @ pixels) * depth.reshape(batch, 1, -1)
    proj = k @ (transform[:, :3, :3] @ cam + transform[:, :3, 3:])
    z = proj[:, 2]
    z_safe = z.clamp(min=1e-3)
    x_norm = 2.0 * (proj[:, 0] / z_safe) / max(width - 1, 1) - 1.0
    y_norm = 2.0 * (proj[:, 1] / z_safe) / max(height - 1, 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).reshape(batch, height, width, 2)
    in_bounds = grid.abs().amax(dim=-1) <= 1.0
    in_front = (z > 1e-3).reshape(batch, height, width)
    valid = (in_bounds & in_front).unsqueeze(1).float()
    computed_depth = z.reshape(batch, 1, height, width)
    return grid, valid, computed_depth


def self_discovered_mask(computed_depth: torch.Tensor, projected_depth: torch.Tensor) -> torch.Tensor:
    """SC-SfMLearner self-discovered mask ``1 - |c - p| / (c + p)``, clamped to ``[0, 1]``.

    ``computed_depth`` is the Z of the target point in the source camera and
    ``projected_depth`` is the source depth sampled at the projected pixel.
    """
    diff = (computed_depth - projected_depth).abs() / (computed_depth + projected_depth).clamp(min=1e-6)
    return (1.0 - diff).clamp(0.0, 1.0)


def sample(
    src: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    return F.grid_sample(src.float(), grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)


def inverse_warp(
    src: torch.Tensor,
    tgt_depth: torch.Tensor,
    src_from_tgt: torch.Tensor,
    intrinsics: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample ``src`` into the target view. Returns ``(warped, valid)``."""
    grid, valid, _ = reprojection_grid(tgt_depth, src_from_tgt, intrinsics)
    return sample(src, grid, mode, padding_mode, align_corners), valid
