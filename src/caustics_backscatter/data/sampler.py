"""Sampling which frames of a sequence form a training clip."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from caustics_backscatter.geometry.chain import build_chain, max_chain_length


@dataclass(frozen=True)
class ClipSampler:
    """Samples ``num_views`` frames with consecutive gaps uniform in ``[gap_min, gap_max]``.

    With ``shuffle`` the view order is randomly permuted, so view 0 is equally likely to be
    before or after view 1 in time. Views are model inputs / loss targets according to the
    task's ``views.inputs`` and ``views.targets``.
    """

    num_views: int
    gap_min: int
    gap_max: int
    max_hop: int | None
    shuffle: bool = True

    @property
    def max_span(self) -> int:
        return (self.num_views - 1) * self.gap_max

    @property
    def max_chain(self) -> int:
        return max_chain_length(self.num_views, self.gap_max, self.max_hop)

    def offsets(self, rng: np.random.Generator) -> list[int]:
        gaps = rng.integers(self.gap_min, self.gap_max + 1, size=self.num_views - 1)
        offsets = [0, *np.cumsum(gaps).tolist()]
        if self.shuffle:
            offsets = [offsets[i] for i in rng.permutation(self.num_views)]
        return offsets

    def chain(self, view_frames: list[int]) -> tuple[list[int], list[int]]:
        return build_chain(view_frames, self.max_hop)
