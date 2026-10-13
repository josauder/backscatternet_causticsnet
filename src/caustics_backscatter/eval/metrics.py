"""Exact F1 / accuracy at every threshold from per-split score histograms.

Scores are binned into ``num_bins`` equal bins over ``[lo, hi)`` (values outside are put in
the first/last bin). Threshold ``k`` predicts "caustics" for every pixel in bins ``>= k``,
i.e. score ``>= lo + k * (hi - lo) / num_bins``; ``k = num_bins`` predicts nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch

# Test-split groups of Table 1 in the paper.
TABLE1_GROUPS: dict[str, tuple[int, ...]] = {
    "1-7": (1, 2, 3, 4, 5, 6, 7),
    "1,2,3": (1, 2, 3),
    "1,2": (1, 2),
    "5,7": (5, 7),
    "4,6": (4, 6),
    "1": (1,),
    "3": (3,),
    "6": (6,),
}


def bin_scores(scores: torch.Tensor, num_bins: int, lo: float, hi: float) -> torch.Tensor:
    width = (hi - lo) / num_bins
    return ((scores - lo) / width).floor().long().clamp(0, num_bins - 1)


def pos_neg_histograms(bins: torch.Tensor, gt: torch.Tensor, num_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Counts per bin of the ``gt`` (positive) and ``~gt`` (negative) entries of ``bins``.

    Uses a sort (``torch.unique``) instead of ``torch.bincount``: on CUDA, int64 bincount relies
    on compare-and-swap atomics that serialize when most pixels fall into the same bin (as for
    caustics-free pixels with ``dL ~ 0``), taking seconds per megapixel image.
    """
    keys = bins.flatten() * 2 + gt.flatten().long()
    values, counts = torch.unique(keys, return_counts=True)
    hist = torch.zeros(2 * num_bins, dtype=torch.long, device=bins.device)
    hist[values] = counts
    hist = hist.view(num_bins, 2)
    return hist[:, 1], hist[:, 0]


def threshold_values(num_bins: int, lo: float, hi: float) -> torch.Tensor:
    return lo + torch.arange(num_bins + 1, dtype=torch.float64) * (hi - lo) / num_bins


def confusion_curves(pos_hist: torch.Tensor, neg_hist: torch.Tensor) -> dict[str, torch.Tensor]:
    """TP/FP/FN/TN for each of the ``num_bins + 1`` thresholds. Histograms ``[..., num_bins]``."""
    pos = pos_hist.double()
    neg = neg_hist.double()
    zero = torch.zeros_like(pos[..., :1])
    tp = torch.cat([pos.flip(-1).cumsum(-1).flip(-1), zero], dim=-1)
    fp = torch.cat([neg.flip(-1).cumsum(-1).flip(-1), zero], dim=-1)
    fn = pos.sum(-1, keepdim=True) - tp
    tn = neg.sum(-1, keepdim=True) - fp
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def f1_accuracy(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1.0)
    acc = (tp + tn) / (tp + fp + fn + tn).clamp(min=1.0)
    return f1, acc


def rcaustics_metrics(
    pos_hist: Mapping[int, torch.Tensor],
    neg_hist: Mapping[int, torch.Tensor],
    num_bins: int,
    lo: float,
    hi: float,
    groups: Mapping[str, Iterable[int]] = TABLE1_GROUPS,
) -> dict[str, float]:
    """F1 (caustics class) and accuracy per split group, for one global best threshold and for
    best thresholds chosen per split (the two settings of Table 1)."""
    splits = sorted(pos_hist)
    curves = {s: confusion_curves(pos_hist[s], neg_hist[s]) for s in splits}
    thresholds = threshold_values(num_bins, lo, hi)

    pooled = {key: sum(curves[s][key] for s in splits) for key in ("tp", "fp", "fn", "tn")}
    global_k = int(f1_accuracy(**pooled)[0].argmax())
    split_k = {s: int(f1_accuracy(**curves[s])[0].argmax()) for s in splits}

    out: dict[str, float] = {"global_threshold": float(thresholds[global_k])}
    for s in splits:
        out[f"per_split_threshold/{s}"] = float(thresholds[split_k[s]])
    for name, members in groups.items():
        members = [s for s in members if s in curves]
        if not members:
            continue
        for setting, pick in (("global", lambda s: global_k), ("per_split", lambda s: split_k[s])):
            counts = {key: sum(curves[s][key][pick(s)] for s in members) for key in ("tp", "fp", "fn", "tn")}
            f1, acc = f1_accuracy(**counts)
            out[f"{setting}/f1/{name}"] = float(f1) * 100.0
            out[f"{setting}/acc/{name}"] = float(acc) * 100.0
    return out
