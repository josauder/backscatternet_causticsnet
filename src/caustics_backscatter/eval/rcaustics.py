"""Zero-shot caustics segmentation on R-CAUSTIC (Agrafiotis et al. 2023), Section 4.2 of the paper.

Each image is resized to the CausticsNet input size, the predicted caustics are upsampled
bilinearly to full resolution, and the score ``dL = L(I) - L(clip(I - C))`` (LAB lightness
difference between the input and the cleaned image) is thresholded and compared with the
ground-truth masks. R-CAUSTIC only annotates positive (bright) caustics, so large ``dL``
means caustics.

Expected layout (R-CAUSTIC v2.0 zip): ``.../dataset_<k>_<view>/subset_<k>_<view>_<n>.JPG``
with ground truth ``subset_<k>_<view>_<n>.bin.JPG`` next to it; ``k`` (1-7) is the split.
"""

from __future__ import annotations

import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from kornia.color import rgb_to_lab
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_jpeg, read_file

from caustics_backscatter.eval.metrics import TABLE1_GROUPS, bin_scores, pos_neg_histograms, rcaustics_metrics
from caustics_backscatter.registry import EVALUATORS
from caustics_backscatter.utils.viz import mask_to_uint8, signed_to_uint8, to_uint8

_MASK_RE = re.compile(r"^(?P<stem>.+)\.bin\.(jpe?g|png)$", re.IGNORECASE)
_SPLIT_RE = re.compile(r"subset_(\d+)_", re.IGNORECASE)
_IMAGE_EXTS = (".JPG", ".jpg", ".jpeg", ".JPEG", ".png", ".PNG")


@dataclass(frozen=True)
class RCausticsSample:
    image: Path
    mask: Path
    split: int


def find_rcaustics_samples(root: str | Path) -> list[RCausticsSample]:
    samples: dict[str, RCausticsSample] = {}
    for mask in sorted(Path(root).rglob("*")):
        match = _MASK_RE.match(mask.name)
        if not match or not mask.is_file():
            continue
        stem = match.group("stem")
        split = _SPLIT_RE.search(stem)
        image = next((mask.parent / f"{stem}{ext}" for ext in _IMAGE_EXTS if (mask.parent / f"{stem}{ext}").exists()), None)
        if split is None or image is None or stem in samples:
            continue
        samples[stem] = RCausticsSample(image=image, mask=mask, split=int(split.group(1)))
    return sorted(samples.values(), key=lambda s: (s.split, s.image.name))


class _EncodedFiles(Dataset):
    """Reads encoded bytes in workers; decoding happens on the GPU (nvjpeg) when available."""

    def __init__(self, samples: list[RCausticsSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        # read_file returns an mmapped storage whose fd is already closed; sending it to the main
        # process with the Linux "file_descriptor" sharing strategy fails (EBADF in os.dup) inside
        # the worker's queue feeder thread, and the main process then waits forever.
        return {
            "index": index,
            "split": sample.split,
            "image": read_file(str(sample.image)).clone(),
            "mask": read_file(str(sample.mask)).clone(),
        }


def _decode(data: torch.Tensor, mode: ImageReadMode, device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        try:
            return decode_jpeg(data, mode=mode, device=device)
        except RuntimeError:
            pass
    return decode_jpeg(data, mode=mode).to(device)


def _log(message: str) -> None:
    print(f"[rcaustics] {message}", file=sys.stderr, flush=True)


class _StageTimer:
    """Wall time per stage. Synchronizes CUDA at every lap so GPU work is charged to its stage."""

    def __init__(self, device: torch.device) -> None:
        self.sync = device.type == "cuda"
        self.last: dict[str, float] = {}
        self.window: dict[str, float] = defaultdict(float)
        self._t = time.perf_counter()

    def lap(self, name: str) -> None:
        if self.sync:
            torch.cuda.synchronize()
        now = time.perf_counter()
        self.last[name] = now - self._t
        self.window[name] += now - self._t
        self._t = now

    @staticmethod
    def format(times: dict[str, float], count: int = 1) -> str:
        return ", ".join(f"{name} {1000 * t / count:.0f} ms" for name, t in times.items())


@EVALUATORS.register("rcaustics")
class RCausticsEvaluator:
    name = "rcaustics"

    def __init__(
        self,
        root: str,
        input_size: tuple[int, int] = (256, 512),
        num_bins: int = 4000,
        dl_range: tuple[float, float] = (-100.0, 100.0),
        num_workers: int = 8,
        max_images: int | None = None,
        num_viz: int = 7,
        viz_width: int = 640,
        groups: dict[str, list[int]] | None = None,
        log_every: int = 25,
        log_first: int = 3,
        loader_timeout: float = 600.0,
    ) -> None:
        self.root = Path(root)
        self.loader_timeout = loader_timeout
        self.log_every = log_every
        self.log_first = log_first
        self.input_size = tuple(input_size)
        self.num_bins = num_bins
        self.lo, self.hi = map(float, dl_range)
        self.num_workers = num_workers
        self.num_viz = num_viz
        self.viz_width = viz_width
        self.groups = groups or TABLE1_GROUPS
        self.samples = find_rcaustics_samples(self.root) if self.root.exists() else []
        if max_images is not None:
            self.samples = self.samples[:: max(1, len(self.samples) // max_images)][:max_images]
        if not self.samples:
            _log(f"no samples found under {self.root}; evaluator disabled")
        else:
            splits = sorted({s.split for s in self.samples})
            _log(f"{len(self.samples)} images in splits {splits}")
        # One visualization image per split (the first of each), up to num_viz.
        first_per_split: dict[int, int] = {}
        for i, sample in enumerate(self.samples):
            first_per_split.setdefault(sample.split, i)
        self.viz_indices = set(list(first_per_split.values())[:num_viz])

    @torch.no_grad()
    def __call__(self, task) -> tuple[dict[str, float], dict[str, list[np.ndarray]]]:
        if not self.samples:
            return {}, {}
        device = task.device
        pos: dict[int, torch.Tensor] = {}
        neg: dict[int, torch.Tensor] = {}
        viz: list[dict] = []
        total = len(self.samples)
        _log(f"start: {total} images, {self.num_workers} loader workers, device {device}")
        loader = DataLoader(
            _EncodedFiles(self.samples),
            batch_size=None,
            num_workers=self.num_workers,
            timeout=self.loader_timeout if self.num_workers > 0 else 0,
        )
        start = time.perf_counter()
        window_start, window_count = start, 0
        timer = _StageTimer(device)
        for done, item in enumerate(loader, start=1):
            timer.lap("read")
            if done == 1:
                _log(f"first image after {timer.last['read']:.1f} s (loader startup + file read)")
            image = _decode(item["image"], ImageReadMode.RGB, device).float().div_(255.0)[None]
            mask = _decode(item["mask"], ImageReadMode.GRAY, device)[None].float()
            height, width = image.shape[-2:]
            if mask.shape[-2:] != (height, width):
                mask = F.interpolate(mask, size=(height, width), mode="nearest")
            gt = mask[0, 0] > 127
            timer.lap("decode")

            small = F.interpolate(image, size=self.input_size, mode="bilinear", antialias=True, align_corners=False)
            caustics = task.predict_caustics(small).float()
            timer.lap("predict")
            caustics_full = F.interpolate(caustics, size=(height, width), mode="bilinear", align_corners=False)
            cleaned = (image - caustics_full).clamp(0.0, 1.0)
            score = rgb_to_lab(image)[0, 0] - rgb_to_lab(cleaned)[0, 0]
            timer.lap("score")

            bins = bin_scores(score, self.num_bins, self.lo, self.hi)
            split = int(item["split"])
            if split not in pos:
                pos[split] = torch.zeros(self.num_bins, dtype=torch.long, device=device)
                neg[split] = torch.zeros(self.num_bins, dtype=torch.long, device=device)
            num_pos, num_neg = pos_neg_histograms(bins, gt, self.num_bins)
            pos[split] += num_pos
            neg[split] += num_neg
            timer.lap("hist")

            if int(item["index"]) in self.viz_indices:
                size = (round(height * self.viz_width / width), self.viz_width)
                viz.append(
                    {
                        "rgb": F.interpolate(image, size=size, mode="area")[0],
                        "caustics": F.interpolate(caustics_full, size=size, mode="area")[0],
                        "score": F.interpolate(score[None, None], size=size, mode="area")[0, 0],
                        "gt": F.interpolate(gt[None, None].float(), size=size, mode="area")[0, 0] > 0.5,
                    }
                )
                timer.lap("viz")

            window_count += 1
            if done <= self.log_first:
                name = self.samples[int(item["index"])].image.name
                _log(f"image {done}/{total} {name} ({width}x{height}, split {split}): {timer.format(timer.last)}")
            if self.log_every and (done % self.log_every == 0 or done == total):
                now = time.perf_counter()
                elapsed = now - start
                eta = elapsed / done * (total - done)
                _log(
                    f"{done}/{total} images | {(now - window_start) / window_count:.3f} s/image (last {window_count})"
                    f" | elapsed {elapsed / 60:.1f} min | ETA {eta / 60:.1f} min"
                    f" | per image: {timer.format(timer.window, window_count)}"
                )
                timer.window.clear()
                window_start, window_count = now, 0

        metrics = rcaustics_metrics(
            {s: h.cpu() for s, h in pos.items()},
            {s: h.cpu() for s, h in neg.items()},
            self.num_bins,
            self.lo,
            self.hi,
            self.groups,
        )
        for split in pos:
            num_pos = float(pos[split].sum())
            metrics[f"gt_positive_frac/{split}"] = num_pos / max(num_pos + float(neg[split].sum()), 1.0)
        threshold = metrics["global_threshold"]
        _log(
            f"done: {total} images in {(time.perf_counter() - start) / 60:.1f} min; "
            f"global F1 (1-7) {metrics.get('global/f1/1-7', float('nan')):.2f} at dL >= {threshold:.2f}"
        )
        gain = task.cfg.viz.residual_gain
        panels = {
            "rgb": [to_uint8(v["rgb"]) for v in viz],
            "caustics": [signed_to_uint8(v["caustics"], gain) for v in viz],
            "pred_mask": [mask_to_uint8(v["score"] >= threshold) for v in viz],
            "gt_mask": [mask_to_uint8(v["gt"]) for v in viz],
        }
        return {f"{self.name}/{k}": v for k, v in metrics.items()}, panels
