import numpy as np
import torch

from caustics_backscatter.data.sampler import ClipSampler
from caustics_backscatter.geometry.base import Geometry
from caustics_backscatter.geometry.warp import self_discovered_mask
from caustics_backscatter.losses import (
    clip_loss,
    far_backscatter_loss,
    l2,
    masked_mean,
    photometric_map,
    stationary_automask,
    tv_squared,
)
from caustics_backscatter.losses.backscatternet import backscatternet_loss
from caustics_backscatter.losses.causticsnet import causticsnet_loss, view_pairs
from caustics_backscatter.models.backscatternet import backscatter

H, W = 16, 24
SHIFT = 4
DELTA = 0.02
FX = 20.0


def identity_geometry(bsz: int, views: int = 2) -> Geometry:
    k = torch.tensor([[20.0, 0.0, (W - 1) / 2], [0.0, 20.0, (H - 1) / 2], [0.0, 0.0, 1.0]]).repeat(bsz, 1, 1)
    return Geometry(
        depth=torch.ones(bsz, views, 1, H, W),
        T=torch.eye(4).expand(bsz, views, views, 4, 4).clone(),
        K=k,
    )


def test_regularizers():
    assert tv_squared(torch.full((2, 3, H, W), 0.7)) == 0
    ramp = torch.arange(W, dtype=torch.float32).expand(1, 1, H, W) * 0.1
    assert torch.isclose(tv_squared(ramp), torch.tensor(0.01))
    assert torch.isclose(l2(torch.full((1, 3, 4, 4), 0.5)), torch.tensor(0.25))
    assert clip_loss(torch.rand(2, 3, H, W)) == 0
    assert torch.isclose(clip_loss(torch.tensor([1.5, -0.5, 0.5, 0.5])), torch.tensor(0.125))


def test_photometric_and_masked_mean():
    a = torch.rand(2, 3, H, W)
    assert photometric_map(a, a).abs().max() < 1e-6
    values = torch.ones(2, 3, H, W)
    values[:, :, :, : W // 2] = 5.0
    weight = torch.zeros(2, 1, H, W)
    weight[..., W // 2 :] = 1.0
    assert torch.isclose(masked_mean(values, weight), torch.tensor(1.0))


def test_far_backscatter_loss_zero_when_matching_far_color():
    image = torch.rand(2, 3, H, W)
    depth = torch.rand(2, 1, H, W) + 0.1
    far = depth > torch.quantile(depth.flatten(1), 0.95, dim=1).view(-1, 1, 1, 1)
    mean_color = (image * far).sum((2, 3), keepdim=True) / far.sum((2, 3), keepdim=True)
    assert far_backscatter_loss(mean_color.expand_as(image), image, depth) < 1e-10
    assert far_backscatter_loss(torch.zeros_like(image), image, depth) > 0


def test_view_pairs():
    assert view_pairs([0, 1], [0, 1]) == [(0, 1), (1, 0)]
    assert view_pairs([0, 1], [2]) == [(0, 2), (1, 2)]


def shift_pair(depth0: float = 1.0, depth1: float = 1.0, delta: float = DELTA):
    """View 1 is view 0 shifted right by ``SHIFT`` plus ``delta``.

    The poses shift by that many pixels at the given per-view depths, so both warp
    directions recover a constant residual of ``delta`` and the auto-mask stays on.
    """
    xs = torch.arange(W, dtype=torch.float32)
    ramp = (0.25 + 0.5 * xs / (W - 1)).view(1, 1, 1, 1, W).expand(2, 1, 3, H, W).contiguous()
    view1 = torch.zeros_like(ramp)
    view1[..., SHIFT:] = ramp[..., :-SHIFT] + delta
    images = torch.cat([ramp, view1], dim=1)
    k = torch.tensor([[FX, 0.0, (W - 1) / 2], [0.0, FX, (H - 1) / 2], [0.0, 0.0, 1.0]]).repeat(2, 1, 1)
    depth = torch.ones(2, 2, 1, H, W)
    depth[:, 0] = depth0
    depth[:, 1] = depth1
    transforms = torch.eye(4).repeat(2, 2, 2, 1, 1)
    transforms[:, 1, 0, 0, 3] = SHIFT * depth0 / FX
    transforms[:, 0, 1, 0, 3] = -SHIFT * depth1 / FX
    prediction = torch.zeros(2, 2, 3, H, W)
    prediction[:, 0] = -delta
    prediction[:, 1] = delta
    return images, Geometry(depth=depth, T=transforms, K=k), prediction


def test_self_discovered_mask_identity_and_mismatch():
    ones = torch.ones(1, 1, 2, 2)
    assert torch.allclose(self_discovered_mask(ones, ones), ones)
    assert torch.allclose(self_discovered_mask(ones, torch.zeros_like(ones)), torch.zeros_like(ones))
    assert torch.allclose(self_discovered_mask(torch.full_like(ones, 3.0), ones), torch.full_like(ones, 0.5))


def test_stationary_automask_keep_and_drop():
    target = torch.zeros(1, 3, 2, 2)
    source = torch.full_like(target, 0.4)
    assert stationary_automask(target, torch.full_like(target, 0.1), source).eq(1).all()
    assert stationary_automask(target, source, source).eq(0).all()
    assert stationary_automask(target, torch.full_like(target, 0.8), source).eq(0).all()


def test_causticsnet_residual_is_a_detached_target():
    images, geom, _ = shift_pair()
    images = images.requires_grad_(True)
    prediction = torch.zeros(2, 2, 3, H, W, requires_grad=True)
    weights = torch.ones(2, 2, 1, H, W)
    weights[:, 0, :, 0, 0] = 0
    total, logs, residuals = causticsnet_loss(
        prediction, images, geom, weights, [0, 1], [0, 1], tv=0, l2_weight=0, clip=0, align_corners=True
    )
    assert residuals[0].grad_fn is None
    assert torch.all(residuals[0][..., 0, 0] == 0)
    assert logs["data"] > 0
    total.backward()
    assert prediction.grad is not None and prediction.grad.abs().sum() > 0
    # A detached residual sends no gradient into the frames. The zero-weighted clip
    # term still references them, so their gradient is present and identically zero.
    assert images.grad is not None and torch.count_nonzero(images.grad) == 0


def test_causticsnet_loss_perfect_prediction():
    images, geom, prediction = shift_pair()
    weights = torch.ones(2, 2, 1, H, W)
    total, logs, residuals = causticsnet_loss(prediction, images, geom, weights, [0, 1], [0, 1], align_corners=True)
    assert logs["data"] < 1e-6
    assert logs["mask_mean"] > 0.99
    assert torch.allclose(residuals[0][:, :, 8, 8], torch.full((2, 3), -DELTA), atol=1e-5)
    assert torch.isclose(total, logs["tv"] + 0.05 * logs["l2"] + logs["clip"])
    _, wrong, _ = causticsnet_loss(torch.zeros_like(prediction), images, geom, weights, [0, 1], [0, 1], align_corners=True)
    assert wrong["data"] > 1e-4


def test_consistency_mask_scales_pair_loss():
    weights = torch.ones(2, 2, 1, H, W)
    images, matched, _ = shift_pair(1.0, 1.0)
    images_m, mismatched, _ = shift_pair(3.0, 1.0)
    zeros = torch.zeros(2, 2, 3, H, W)
    _, full, _ = causticsnet_loss(zeros, images, matched, weights, [0, 1], [0, 1], align_corners=True)
    _, half, _ = causticsnet_loss(zeros, images_m, mismatched, weights, [0, 1], [0, 1], align_corners=True)
    assert full["data"] > 1e-4
    assert torch.isclose(half["mask_mean"], torch.tensor(0.5), atol=1e-4)
    assert torch.isclose(half["data"], full["data"] * 0.5, rtol=1e-4)

    _, photo_full = backscatternet_loss(
        images, zeros, images, matched, weights, [0, 1], [0, 1], style=0, align_corners=True
    )
    _, photo_half = backscatternet_loss(
        images_m, zeros, images_m, mismatched, weights, [0, 1], [0, 1], style=0, align_corners=True
    )
    assert torch.isclose(photo_half["photometric"], photo_full["photometric"] * 0.5, rtol=1e-4)


def test_causticsnet_loss_source_mask_zeroes_target():
    images, geom, residual = shift_pair()
    patch = (..., slice(4, 8), slice(6, 12))

    # Masked in the target view: the target is zero there, and the pixel is still supervised.
    weights = torch.ones(2, 2, 1, H, W)
    weights[:, 0][patch] = 0
    _, logs, residuals = causticsnet_loss(residual, images, geom, weights, [0, 1], [0, 1], align_corners=True)
    assert torch.all(residuals[0][patch] == 0)
    zeroed = residual.clone()
    zeroed[:, 0][patch] = 0
    _, logs_zeroed, _ = causticsnet_loss(zeroed, images, geom, weights, [0, 1], [0, 1], align_corners=True)
    assert logs["data"] > logs_zeroed["data"]
    assert logs_zeroed["data"] < 1e-6

    # Masked in the reprojected view: the shifted pixel has zero weight, so its prediction is ignored.
    weights = torch.ones(2, 2, 1, H, W)
    weights[:, 1][patch] = 0
    _, logs_a, _ = causticsnet_loss(residual, images, geom, weights, [0, 1], [0, 1], align_corners=True)
    perturbed = residual.clone()
    perturbed[:, 0][..., 4:8, 2:8] += 0.3
    _, logs_b, _ = causticsnet_loss(perturbed, images, geom, weights, [0, 1], [0, 1], align_corners=True)
    assert torch.isclose(logs_a["data"], logs_b["data"])


def test_shuffle_draws_both_temporal_orders():
    rng = np.random.default_rng(0)
    shuffled = ClipSampler(num_views=2, gap_min=3, gap_max=3, max_hop=3, shuffle=True)
    offsets = [shuffled.offsets(rng) for _ in range(80)]
    assert any(left < right for left, right in offsets)
    assert any(left > right for left, right in offsets)
    ordered = ClipSampler(num_views=2, gap_min=3, gap_max=3, max_hop=3, shuffle=False)
    assert all(left < right for left, right in (ordered.offsets(rng) for _ in range(10)))


def test_backscatternet_loss_terms():
    torch.manual_seed(0)
    images = torch.rand(2, 2, 3, H, W)
    geom = identity_geometry(2)
    b_inf = torch.full((2, 2, 3), 0.2)
    beta = torch.full((2, 2, 3), 1.0)
    scatter = backscatter(b_inf, beta, geom.depth)
    assert scatter.shape == images.shape
    descattered = images - scatter
    total, logs = backscatternet_loss(
        descattered, scatter, images, geom, torch.ones(2, 2, 1, H, W), [0, 1], [0, 1],
        descattered_jittered=descattered.clone(), align_corners=True,
    )
    assert logs["style"] < 1e-6
    assert torch.isfinite(total)
