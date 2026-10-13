import math

import pytest
import torch

from caustics_backscatter.data.augment import hflip_images
from caustics_backscatter.geometry.base import Geometry
from caustics_backscatter.geometry.chain import build_chain, hop_sizes, max_chain_length
from caustics_backscatter.geometry.pose import compose_chain, invert_se3, relative_poses
from caustics_backscatter.geometry.warp import inverse_warp, reprojection_grid

H, W = 24, 40


def random_se3(n: int, rot_scale: float = 0.05, trans_scale: float = 0.1) -> torch.Tensor:
    axis_angle = torch.randn(n, 3) * rot_scale
    angle = axis_angle.norm(dim=1, keepdim=True).clamp(min=1e-8)
    k = axis_angle / angle
    kx = torch.zeros(n, 3, 3)
    kx[:, 0, 1], kx[:, 0, 2] = -k[:, 2], k[:, 1]
    kx[:, 1, 0], kx[:, 1, 2] = k[:, 2], -k[:, 0]
    kx[:, 2, 0], kx[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = torch.sin(angle)[..., None], torch.cos(angle)[..., None]
    rot = torch.eye(3) + s * kx + (1 - c) * kx @ kx
    out = torch.eye(4).repeat(n, 1, 1)
    out[:, :3, :3] = rot
    out[:, :3, 3] = torch.randn(n, 3) * trans_scale
    return out


def intrinsics(n: int) -> torch.Tensor:
    k = torch.tensor([[30.0, 0.0, 19.0], [0.0, 30.0, 11.0], [0.0, 0.0, 1.0]])
    return k.repeat(n, 1, 1)


def smooth_image(n: int) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
    base = torch.stack([torch.sin(3 * xs + ys), torch.cos(2 * ys - xs), xs * ys], dim=0)
    return (0.5 + 0.4 * base).unsqueeze(0).repeat(n, 1, 1, 1) + 0.01 * torch.rand(n, 3, H, W)


def test_identity_reprojection_depth_is_target_depth():
    depth = 1.0 + torch.rand(2, 1, H, W)
    _, valid, computed = reprojection_grid(depth, torch.eye(4).repeat(2, 1, 1), intrinsics(2))
    assert torch.allclose(computed, depth, atol=1e-5)
    assert valid[..., 1:-1, 1:-1].min() == 1.0


def test_identity_warp_returns_input():
    torch.manual_seed(0)
    img = torch.rand(2, 3, H, W)
    depth = 1.0 + torch.rand(2, 1, H, W)
    warped, valid = inverse_warp(img, depth, torch.eye(4).repeat(2, 1, 1), intrinsics(2), align_corners=True)
    assert torch.allclose(warped, img, atol=1e-4)
    assert valid[..., 1:-1, 1:-1].min() == 1.0  # border pixels sit at |grid| = 1 up to rounding


def test_identity_warp_deepreefmap_convention_is_close():
    # (W-1) normalization with align_corners=False slightly rescales around the center.
    img = smooth_image(1)
    warped, _ = inverse_warp(img, torch.ones(1, 1, H, W), torch.eye(4)[None], intrinsics(1), align_corners=False)
    assert (warped - img)[..., 2:-2, 2:-2].abs().max() < 0.05


@pytest.mark.parametrize("align_corners", [False, True])
def test_warp_commutes_with_flip(align_corners):
    torch.manual_seed(1)
    bsz = 3
    src = torch.rand(bsz, 3, H, W)
    depth = 2.0 + torch.rand(bsz, 1, H, W)
    transform = random_se3(bsz)
    k = intrinsics(bsz)
    warped, valid = inverse_warp(src, depth, transform, k, align_corners=align_corners)

    flip = torch.ones(bsz, dtype=torch.bool)
    geom = Geometry(depth=depth[:, None], T=transform[:, None, None], K=k).hflip(flip)
    warped_f, valid_f = inverse_warp(src.flip(-1), geom.depth[:, 0], geom.T[:, 0, 0], geom.K, align_corners=align_corners)
    assert torch.allclose(warped_f, warped.flip(-1), atol=1e-4)
    assert torch.equal(valid_f, valid.flip(-1))


def test_hflip_only_affects_selected_samples():
    depth = torch.rand(2, 2, 1, H, W)
    transforms = random_se3(8).view(2, 2, 2, 4, 4)
    geom = Geometry(depth=depth, T=transforms, K=intrinsics(2))
    flipped = geom.hflip(torch.tensor([True, False]))
    assert torch.equal(flipped.depth[1], depth[1])
    assert torch.equal(flipped.T[1], transforms[1])
    assert torch.equal(flipped.depth[0], depth[0].flip(-1))
    assert flipped.K[0, 0, 2] == (W - 1) - geom.K[0, 0, 2]
    images = torch.rand(2, 2, 3, H, W)
    out = hflip_images(images, torch.tensor([True, False]))
    assert torch.equal(out[0], images[0].flip(-1)) and torch.equal(out[1], images[1])


def test_invert_se3():
    t = random_se3(5, rot_scale=0.5, trans_scale=1.0)
    assert torch.allclose(t @ invert_se3(t), torch.eye(4).expand(5, 4, 4), atol=1e-5)


def test_chained_relative_poses():
    torch.manual_seed(2)
    hops = random_se3(4).view(1, 4, 4, 4)  # chain of 5 frames
    chain = compose_chain(hops)
    view_pos = torch.tensor([[4, 0, 2]])
    rel = relative_poses(chain, view_pos)
    t_40 = hops[0, 3] @ hops[0, 2] @ hops[0, 1] @ hops[0, 0]
    t_20 = hops[0, 1] @ hops[0, 0]
    assert torch.allclose(rel[0, 0, 1], t_40, atol=1e-5)
    assert torch.allclose(rel[0, 1, 0], invert_se3(t_40), atol=1e-5)
    assert torch.allclose(rel[0, 0, 2], hops[0, 3] @ hops[0, 2], atol=1e-5)
    assert torch.allclose(rel[0, 2, 1], t_20, atol=1e-5)
    assert torch.allclose(rel[0, 1, 1], torch.eye(4), atol=1e-6)


@pytest.mark.parametrize("gap", range(1, 40))
def test_hop_sizes(gap):
    sizes = hop_sizes(gap, 3)
    assert sum(sizes) == gap
    assert len(sizes) == math.ceil(gap / 3)
    assert max(sizes) <= 3
    if gap >= 3:
        assert min(sizes) >= 2
    assert hop_sizes(gap, None) == [gap]


def test_hop_sizes_example():
    assert sorted(hop_sizes(25, 3)) == [2, 2, 3, 3, 3, 3, 3, 3, 3]


def test_build_chain():
    chain, pos = build_chain([110, 100, 104], max_hop=3)
    assert chain == [100, 102, 104, 107, 110]
    assert [chain[p] for p in pos] == [110, 100, 104]
    assert len(chain) <= max_chain_length(3, 6, 3)
    chain, pos = build_chain([7, 3], max_hop=None)
    assert chain == [3, 7] and pos == [1, 0]
