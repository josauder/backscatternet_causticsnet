"""Geometry interface shared by all depth/pose/intrinsics sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn

from caustics_backscatter.geometry.pose import mirror_x


@dataclass
class Geometry:
    """Per-clip geometry for V views.

    Attributes:
        depth: ``[B, V, 1, H, W]`` depth (distance along the optical axis) of every view.
        T: ``[B, V, V, 4, 4]``; ``T[:, i, j] = T_{i<-j}`` maps points from camera j to camera i.
        K: ``[B, 3, 3]`` linear intrinsics at the image resolution.
    """

    depth: torch.Tensor
    T: torch.Tensor
    K: torch.Tensor

    def hflip(self, flip: torch.Tensor) -> Geometry:
        """Geometry of horizontally flipped images for the samples where ``flip`` is true.

        Under the ``2u/(W-1) - 1`` pixel normalization, flipping maps pixel u to W-1-u,
        so ``cx' = W-1-cx``, camera x-axes are mirrored (``T' = S T S``) and depth maps
        are flipped.
        """
        width = self.depth.shape[-1]
        f_depth = flip.view(-1, 1, 1, 1, 1)
        f_pose = flip.view(-1, 1, 1, 1, 1)
        k_flipped = self.K.clone()
        k_flipped[:, 0, 2] = (width - 1) - self.K[:, 0, 2]
        return Geometry(
            depth=torch.where(f_depth, self.depth.flip(-1), self.depth),
            T=torch.where(f_pose, mirror_x(self.T), self.T),
            K=torch.where(flip.view(-1, 1, 1), k_flipped, self.K),
        )


class GeometryProvider(nn.Module, ABC):
    """Produces :class:`Geometry` for a batch from the clip dataset.

    ``max_hop`` tells the dataset which intermediate frames to load (see
    :mod:`caustics_backscatter.geometry.chain`); providers that do not need intermediate
    frames (e.g. precomputed geometry) set it to ``None``.

    The batch contains ``frames`` ``[B, C, 3, H, W]`` uint8 chain frames, ``chain_valid``
    ``[B, C]``, ``view_pos`` ``[B, V]``, ``frame_ids`` ``[B, C]``, ``seq_idx`` ``[B]`` and ``K``.
    """

    max_hop: int | None = None

    @abstractmethod
    def forward(self, batch: dict[str, torch.Tensor]) -> Geometry: ...

    def train(self, mode: bool = True) -> GeometryProvider:
        # Geometry is a frozen teacher: keep BatchNorm etc. in eval mode.
        return super().train(False)
