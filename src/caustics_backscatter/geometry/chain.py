"""Which frames to load so the pose between views can be chained over short hops.

The SC-SfMLearner pose network was trained on frames 3 apart (20 fps), so the pose
between two views further apart is composed from hops of at most ``max_hop`` frames.
A gap ``g`` is split into ``ceil(g / max_hop)`` hops of near-equal size; with
``max_hop=3`` every hop of a gap >= 3 is 2 or 3 frames (e.g. 25 = 3*7 + 2*2).
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def hop_sizes(gap: int, max_hop: int | None) -> list[int]:
    if gap <= 0:
        raise ValueError(f"gap must be positive, got {gap}")
    if max_hop is None:
        return [gap]
    num_hops = math.ceil(gap / max_hop)
    base, remainder = divmod(gap, num_hops)
    return [base + 1] * remainder + [base] * (num_hops - remainder)


def build_chain(view_frames: Sequence[int], max_hop: int | None) -> tuple[list[int], list[int]]:
    """Return ``(chain_frames, view_positions)``.

    ``chain_frames`` is sorted ascending, contains every view frame plus the intermediate
    hop frames between consecutive views, and ``chain_frames[view_positions[v]] ==
    view_frames[v]``. With ``max_hop=None`` the chain is just the sorted unique views.
    """
    unique = sorted(set(view_frames))
    chain = [unique[0]]
    for start, end in zip(unique[:-1], unique[1:]):
        frame = start
        for size in hop_sizes(end - start, max_hop):
            frame += size
            chain.append(frame)
    position = {frame: i for i, frame in enumerate(chain)}
    return chain, [position[f] for f in view_frames]


def max_chain_length(num_views: int, gap_max: int, max_hop: int | None) -> int:
    hops_per_gap = 1 if max_hop is None else math.ceil(gap_max / max_hop)
    return (num_views - 1) * hops_per_gap + 1
