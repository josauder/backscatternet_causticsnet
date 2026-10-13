"""Rigid transforms as 4x4 matrices. ``T_{a<-b}`` maps points from camera b into camera a."""

from __future__ import annotations

import torch


def invert_se3(transform: torch.Tensor) -> torch.Tensor:
    rot = transform[..., :3, :3]
    trans = transform[..., :3, 3:]
    rot_t = rot.transpose(-1, -2)
    out = torch.zeros_like(transform)
    out[..., :3, :3] = rot_t
    out[..., :3, 3:] = -rot_t @ trans
    out[..., 3, 3] = 1.0
    return out


def compose_chain(hops: torch.Tensor) -> torch.Tensor:
    """Accumulate hop transforms along a frame chain.

    ``hops[:, k] = T_{k+1<-k}`` for a chain of C frames ``[B, C-1, 4, 4]``. Returns
    ``P[:, k] = T_{k<-0}`` with shape ``[B, C, 4, 4]``.
    """
    batch, num_hops = hops.shape[:2]
    eye = torch.eye(4, dtype=hops.dtype, device=hops.device).expand(batch, 4, 4)
    poses = [eye]
    for k in range(num_hops):
        poses.append(hops[:, k] @ poses[-1])
    return torch.stack(poses, dim=1)


def relative_poses(chain_poses: torch.Tensor, view_pos: torch.Tensor) -> torch.Tensor:
    """Pairwise transforms between views.

    ``chain_poses[:, k] = T_{k<-0}`` and ``view_pos[b, v]`` is the chain position of view v.
    Returns ``T[b, i, j] = T_{i<-j}`` with shape ``[B, V, V, 4, 4]``. The reverse direction
    is the inverse of the forward composition.
    """
    batch = chain_poses.shape[0]
    views = chain_poses[torch.arange(batch, device=chain_poses.device)[:, None], view_pos]
    return views[:, :, None] @ invert_se3(views)[:, None, :]


def mirror_x(transform: torch.Tensor) -> torch.Tensor:
    """Express a transform in horizontally mirrored cameras: ``S T S`` with ``S = diag(-1, 1, 1, 1)``."""
    sign = torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=transform.dtype, device=transform.device)
    return transform * sign[:, None] * sign[None, :]
