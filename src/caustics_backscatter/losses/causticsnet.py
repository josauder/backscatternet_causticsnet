"""CausticsNet objective (Eq. 6-8)."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from caustics_backscatter.geometry.base import Geometry
from caustics_backscatter.geometry.warp import reprojection_grid, sample, self_discovered_mask
from caustics_backscatter.losses.photometric import masked_mean, stationary_automask
from caustics_backscatter.losses.regularizers import clip_loss, l2, tv_squared


def view_pairs(inputs: Sequence[int], targets: Sequence[int]) -> list[tuple[int, int]]:
    return [(i, j) for i in inputs for j in targets if i != j]


def causticsnet_loss(
    caustics: torch.Tensor,
    images: torch.Tensor,
    geometry: Geometry,
    weights: torch.Tensor,
    inputs: Sequence[int],
    targets: Sequence[int],
    tv: float = 1.0,
    l2_weight: float = 0.05,
    clip: float = 1.0,
    align_corners: bool = False,
    padding_mode: str = "zeros",
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[int, torch.Tensor]]:
    """``||C_i - R_i||^2 + tv * TV(C) + l2 * ||C||^2 + clip * L_[0,1](I - C)``.

    ``R_i = I_i - warp(I_j -> i)`` is the reprojection residual of input view i against
    target view j. ``R_i`` is detached, so it is a fixed regression target and the squared
    error updates only the prediction. That squared error is weighted by the
    self-discovered mask ``1 - |Z - D_j| / (Z + D_j)`` and averaged over valid pixels; the
    mask does not scale ``R_i``. Regularizers act on the full prediction. Valid pixels are
    the projection mask times view j's weights reprojected into view i, times the
    stationary auto-mask (warped L1 strictly below unwarped L1). Every (input, target)
    pair contributes and the data terms are averaged. Where view i's own weight is zero
    the target is set to zero before the detach (caustics are pushed to 0 there); that
    path does not include the auto-mask, so stationary pixels are ignored rather than
    driven to a zero caustic.

    Args:
        caustics: ``[B, len(inputs), 3, H, W]`` predictions for the input views.
        images: ``[B, V, 3, H, W]`` jittered views in [0, 1].
        weights: ``[B, V, 1, H, W]`` pixel weights.

    Returns ``(total, logs, residuals)`` where ``residuals[i]`` is the target of input view i
    against its first target view (for visualization).
    """
    data_terms = []
    valid_fracs = []
    mask_means = []
    residuals: dict[int, torch.Tensor] = {}
    for i, j in view_pairs(inputs, targets):
        grid, valid, computed_depth = reprojection_grid(geometry.depth[:, i], geometry.T[:, j, i], geometry.K)
        warped = sample(images[:, j], grid, "bilinear", padding_mode, align_corners)
        warped_weight = sample(weights[:, j], grid, "bilinear", padding_mode, align_corners)
        projected_depth = sample(geometry.depth[:, j], grid, "bilinear", padding_mode, align_corners)
        consistency = self_discovered_mask(computed_depth, projected_depth)
        stationary = stationary_automask(images[:, i], warped, images[:, j])
        weight = valid * warped_weight * stationary
        target = torch.where(weights[:, i] > 0, images[:, i].float() - warped, 0.0).detach()
        prediction = caustics[:, list(inputs).index(i)].float()
        data_terms.append(masked_mean((prediction - target).pow(2) * consistency, weight))
        valid_fracs.append(valid.mean())
        mask_means.append(masked_mean(consistency, weight))
        residuals.setdefault(i, target * valid)

    caustics = caustics.float()
    cleaned = images[:, list(inputs)].float() - caustics
    data = torch.stack(data_terms).mean()
    tv_term = tv_squared(caustics)
    l2_term = l2(caustics)
    clip_term = clip_loss(cleaned)
    total = data + tv * tv_term + l2_weight * l2_term + clip * clip_term
    logs = {
        "loss": total.detach(),
        "data": data.detach(),
        "tv": tv_term.detach(),
        "l2": l2_term.detach(),
        "clip": clip_term.detach(),
        "valid_frac": torch.stack(valid_fracs).mean().detach(),
        "mask_mean": torch.stack(mask_means).mean().detach(),
        "caustics_abs": caustics.abs().mean().detach(),
    }
    return total, logs, residuals
