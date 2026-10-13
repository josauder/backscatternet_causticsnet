"""Per-pixel loss weights computed from the input views.

A weighter maps ``images [B, V, 3, H, W]`` (un-jittered, in [0, 1]) and, if the dataset
provides them (``data.labels``), per-pixel class ids ``labels [B, V, 1, H, W]`` (uint8) to
weights ``[B, V, 1, H, W]``. Losses multiply the target view's weights with the source
view's weights reprojected into the target view, so e.g. a segmentation weighter that
zeroes fish and divers removes them from both ends of every reprojection. CausticsNet is
the exception on the target-view side: instead of zeroing the weight there, it sets the
regression target to zero.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from caustics_backscatter.registry import PIXEL_WEIGHTERS


class PixelWeighter(nn.Module):
    def forward(self, images: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


@PIXEL_WEIGHTERS.register("uniform")
class UniformWeighter(PixelWeighter):
    def __init__(self, **unused: object) -> None:
        # Command-line overrides are merged, so `weighting.name=uniform` keeps the other
        # weighter's options (e.g. zero_classes) in the section; they are ignored.
        super().__init__()

    def forward(self, images: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        return torch.ones_like(images[:, :, :1])


@PIXEL_WEIGHTERS.register("segmentation")
class SegmentationWeighter(PixelWeighter):
    """Weight 0 for pixels whose class id is in ``zero_classes``, 1 elsewhere."""

    def __init__(self, zero_classes: Sequence[int], num_classes: int = 40) -> None:
        super().__init__()
        if any(not 0 <= c < num_classes for c in zero_classes):
            raise ValueError(f"zero_classes {list(zero_classes)} must lie in [0, {num_classes})")
        lut = torch.zeros(num_classes, dtype=torch.bool)
        lut[list(zero_classes)] = True
        self.register_buffer("zero_lut", lut, persistent=False)

    def forward(self, images: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if labels is None:
            raise ValueError("SegmentationWeighter needs per-pixel labels; set data.labels to a segment_frames tag")
        return (~self.zero_lut[labels.long()]).to(images.dtype)
