"""BackscatterNet objective (Eq. 9-11)."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from caustics_backscatter.geometry.base import Geometry
from caustics_backscatter.geometry.warp import reprojection_grid, sample, self_discovered_mask
from caustics_backscatter.losses.backscatter_prior import far_backscatter_loss
from caustics_backscatter.losses.causticsnet import view_pairs
from caustics_backscatter.losses.photometric import masked_mean, photometric_map, stationary_automask
from caustics_backscatter.losses.regularizers import clip_loss


def backscatternet_loss(
    descattered: torch.Tensor,
    backscatter: torch.Tensor,
    images: torch.Tensor,
    geometry: Geometry,
    weights: torch.Tensor,
    inputs: Sequence[int],
    targets: Sequence[int],
    descattered_jittered: torch.Tensor | None = None,
    ssim_weight: float = 0.85,
    style: float = 0.1,
    backscatter_weight: float = 0.5,
    clip: float = 1.0,
    far_quantile: float = 0.95,
    align_corners: bool = False,
    padding_mode: str = "zeros",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """``L_P(D_i, warp(D_j -> i)) + style * L_I + backscatter * L_B + clip * L_[0,1](D)``.

    Each pair's photometric error is multiplied by the self-discovered mask and averaged
    over the projection-valid pixels, the segmentation weights, and the stationary
    auto-mask. The auto-mask compares the descattered views (L1 before SSIM).

    Args:
        descattered, backscatter: ``[B, V, 3, H, W]`` for every view (``D = I - B``).
        images: ``[B, V, 3, H, W]`` jittered views in [0, 1].
        weights: ``[B, V, 1, H, W]`` pixel weights.
        descattered_jittered: ``[B, len(inputs), 3, H, W]`` descattered per-image jittered
            input views for the style-independence loss ``L_I = L_P(f(I), f(Jitter(I)))``.
    """
    inputs = list(inputs)
    photo_terms = []
    mask_means = []
    for i, j in view_pairs(inputs, targets):
        grid, valid, computed_depth = reprojection_grid(geometry.depth[:, i], geometry.T[:, j, i], geometry.K)
        warped = sample(descattered[:, j], grid, "bilinear", padding_mode, align_corners)
        warped_weight = sample(weights[:, j], grid, "bilinear", padding_mode, align_corners)
        projected_depth = sample(geometry.depth[:, j], grid, "bilinear", padding_mode, align_corners)
        consistency = self_discovered_mask(computed_depth, projected_depth)
        stationary = stationary_automask(descattered[:, i], warped, descattered[:, j])
        weight = valid * weights[:, i] * warped_weight * stationary
        photo = photometric_map(descattered[:, i], warped, ssim_weight) * consistency
        photo_terms.append(masked_mean(photo, weight))
        mask_means.append(masked_mean(consistency, weight))
    photometric = torch.stack(photo_terms).mean()

    d_in = descattered[:, inputs].flatten(0, 1).float()
    w_in = weights[:, inputs].flatten(0, 1)
    if descattered_jittered is not None and style > 0:
        style_term = masked_mean(photometric_map(d_in, descattered_jittered.flatten(0, 1), ssim_weight), w_in)
    else:
        style_term = torch.zeros((), device=d_in.device)
    far_term = far_backscatter_loss(
        backscatter[:, inputs].flatten(0, 1).float(),
        images[:, inputs].flatten(0, 1).float(),
        geometry.depth[:, inputs].flatten(0, 1),
        quantile=far_quantile,
        weight=w_in,
    )
    clip_term = clip_loss(d_in)
    total = photometric + style * style_term + backscatter_weight * far_term + clip * clip_term
    logs = {
        "loss": total.detach(),
        "photometric": photometric.detach(),
        "mask_mean": torch.stack(mask_means).mean().detach(),
        "style": style_term.detach(),
        "far_backscatter": far_term.detach(),
        "clip": clip_term.detach(),
    }
    return total, logs
