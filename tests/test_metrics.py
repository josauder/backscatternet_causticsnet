import torch

from caustics_backscatter.eval.metrics import (
    bin_scores,
    confusion_curves,
    f1_accuracy,
    pos_neg_histograms,
    rcaustics_metrics,
    threshold_values,
)

NUM_BINS, LO, HI = 200, -10.0, 10.0


def brute_force(scores: torch.Tensor, gt: torch.Tensor, k: int) -> tuple[float, float]:
    pred = bin_scores(scores, NUM_BINS, LO, HI) >= k
    tp = (pred & gt).sum().item()
    fp = (pred & ~gt).sum().item()
    fn = (~pred & gt).sum().item()
    tn = (~pred & ~gt).sum().item()
    return 2 * tp / max(2 * tp + fp + fn, 1), (tp + tn) / (tp + fp + fn + tn)


def make_split(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    gt = torch.rand(n, generator=g) < 0.2
    scores = torch.randn(n, generator=g) * 3 + gt.float() * 4
    return scores, gt


def histograms(scores: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bins = bin_scores(scores, NUM_BINS, LO, HI)
    return torch.bincount(bins[gt], minlength=NUM_BINS), torch.bincount(bins[~gt], minlength=NUM_BINS)


def test_curves_match_brute_force():
    scores, gt = make_split(5000, 0)
    pos, neg = histograms(scores, gt)
    f1, acc = f1_accuracy(**confusion_curves(pos, neg))
    for k in (0, 50, 100, 137, NUM_BINS):
        bf1, bacc = brute_force(scores, gt, k)
        assert abs(float(f1[k]) - bf1) < 1e-9
        assert abs(float(acc[k]) - bacc) < 1e-9


def test_pos_neg_histograms_match_bincount():
    scores, gt = make_split(5000, 4)
    # Concentrate most entries in a single bin (like caustics-free pixels with dL ~ 0); the
    # tails leave many bins empty.
    scores[:4000] = 0.01
    bins = bin_scores(scores, NUM_BINS, LO, HI).view(50, 100)
    pos, neg = pos_neg_histograms(bins, gt.view(50, 100), NUM_BINS)
    ref_pos, ref_neg = histograms(scores, gt)
    assert pos.dtype == neg.dtype == torch.long
    assert torch.equal(pos, ref_pos)
    assert torch.equal(neg, ref_neg)


def test_thresholds_correspond_to_scores():
    thresholds = threshold_values(NUM_BINS, LO, HI)
    scores = torch.tensor([-0.05, 0.0, 0.05, 0.1])
    k = int((thresholds == 0.0).nonzero())
    assert (bin_scores(scores, NUM_BINS, LO, HI) >= k).tolist() == [False, True, True, True]


def test_rcaustics_metrics_global_and_per_split():
    data = {s: make_split(3000, s) for s in (1, 2, 3)}
    pos, neg = {}, {}
    for s, (scores, gt) in data.items():
        pos[s], neg[s] = histograms(scores, gt)
    out = rcaustics_metrics(pos, neg, NUM_BINS, LO, HI, groups={"1,2": (1, 2), "3": (3,), "all": (1, 2, 3)})

    all_scores = torch.cat([d[0] for d in data.values()])
    all_gt = torch.cat([d[1] for d in data.values()])
    best = max(range(NUM_BINS + 1), key=lambda k: brute_force(all_scores, all_gt, k)[0])
    assert abs(out["global_threshold"] - float(threshold_values(NUM_BINS, LO, HI)[best])) < 1e-9
    bf1, bacc = brute_force(all_scores, all_gt, best)
    assert abs(out["global/f1/all"] - 100 * bf1) < 1e-6
    assert abs(out["global/acc/all"] - 100 * bacc) < 1e-6

    best3 = max(range(NUM_BINS + 1), key=lambda k: brute_force(*data[3], k)[0])
    assert abs(out["per_split/f1/3"] - 100 * brute_force(*data[3], best3)[0]) < 1e-6
    assert out["per_split/f1/all"] >= out["global/f1/all"] - 1e-6
