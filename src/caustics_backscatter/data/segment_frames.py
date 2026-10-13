"""Per-frame CoralscapesV2 class labels for the sequences written by ``prepare_frames``.

For every finished sequence, the segment is decoded again from its source video exactly as
``prepare_frames`` does (same seek, fps resampling and undistortion, from ``meta.json``),
but resized to ``--input-width`` x ``--input-height`` (default 1024x512) so the segmenter
sees real detail instead of upsampled training frames. The frozen DINOv3 ViT-B/16 + DPT
CoralscapesV2 model (39 classes, id 0 = unlabeled) predicts logits at that size; they are
resized bilinearly to the sequence's frame size and argmaxed, and the class ids are
written as uint8 PNGs pixel-aligned with the frames::

    <root>/sequences/<sequence>/labels_<tag>/000000.png ... meta.json

Each sequence is written to a hidden temporary directory that is renamed into place once
``meta.json`` is written, so a labels directory only appears when it is complete; the job
can be re-run and skips labeled sequences. With ``--follow`` it keeps polling for sequences
that ``prepare_frames`` finishes in the meantime, so both jobs can run at the same time.

    python -m caustics_backscatter.data.segment_frames --root data/red_sea_500 [--follow]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from caustics_backscatter.data.dataset import labels_dir
from caustics_backscatter.data.prepare_frames import iter_rectified_frames

HF_REPO = "EPFL-ECEO/coralscapesv2-dinov3-vitb-lora-dpt"
HF_REVISION = "3ef3c6b1b7ffcae2bfd9354fba2a37ff598c2045"
TAG = "coralscapesv2_vitb39"

# Legacy Coralscapes (V1) label set used by the 39-class CoralscapesV2 models.
CORALSCAPES_CLASSES = (
    "unlabeled", "seagrass", "trash", "other coral dead", "other coral bleached", "sand",
    "other coral alive", "human", "transect tools", "fish", "algae covered substrate",
    "other animal", "unknown hard substrate", "background", "dark", "transect line",
    "massive/meandering bleached", "massive/meandering alive", "rubble", "branching bleached",
    "branching dead", "millepora", "branching alive", "massive/meandering dead", "clam",
    "acropora alive", "sea cucumber", "turbinaria", "table acropora alive", "sponge", "anemone",
    "pocillopora alive", "table acropora dead", "meandering bleached", "stylophora alive",
    "sea urchin", "meandering alive", "meandering dead", "crown of thorn", "dead clam",
)  # fmt: skip


class Segmenter(nn.Module):
    """Frozen segmentation model: uint8 RGB frames ``[N, H, W, 3]`` -> class ids ``[N, h, w]``."""

    def __init__(self, model: nn.Module, mean, std, amp: bool = True) -> None:
        super().__init__()
        self.model = model
        self.amp = amp
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> Segmenter:
        return super().train(False)

    @classmethod
    def from_hub(cls, repo: str = HF_REPO, revision: str | None = HF_REVISION, amp: bool = True) -> Segmenter:
        """Load the model with the inference code shipped in the repo (see its model card).

        Building the model instantiates the gated ``facebook/dinov3-vitb16-pretrain-lvd1689m``
        encoder before the merged weights are loaded, so a Hugging Face token with access to
        it is required.
        """
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(repo, revision=revision, allow_patterns=["*.json", "*.py", "*.safetensors"]))
        spec = importlib.util.spec_from_file_location("coralscapes_hub_model", root / "coralscapes_hub_model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.Dinov3DPTSegmenter.from_pretrained(root)
        return cls(model, model.processor.image_mean, model.processor.image_std, amp=amp)

    @torch.no_grad()
    def forward(self, frames: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        x = (frames.permute(0, 3, 1, 2).float() / 255.0 - self.mean) / self.std
        device_type = x.device.type
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=self.amp and device_type == "cuda"):
            logits = self.model(x)
        logits = F.interpolate(logits.float(), size=out_size, mode="bilinear", align_corners=False)
        return logits.argmax(1).to(torch.uint8)


class _FrameBatches(IterableDataset):
    """Decodes whole sequences per DataLoader worker.

    Yields ``{"key", "start", "frames" [n, H, W, 3] uint8}`` batches of one sequence in
    order, then ``{"key", "end": num_frames}``, or ``{"key", "error"}`` if decoding failed.
    """

    def __init__(self, jobs: list[tuple[str, dict]], width: int, height: int, batch_size: int, threads: int) -> None:
        self.jobs = jobs
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.threads = threads

    def __iter__(self):
        info = get_worker_info()
        jobs = self.jobs if info is None else self.jobs[info.id :: info.num_workers]
        for key, meta in jobs:
            count = 0
            buffer: list[np.ndarray] = []
            try:
                frames = iter_rectified_frames(
                    Path(meta["source"]),
                    meta["camera_profile"],
                    float(meta["fps"]),
                    self.threads,
                    float(meta["source_start_s"]),
                    float(meta["source_end_s"]),
                    self.width,
                    self.height,
                )
                for frame in frames:
                    buffer.append(frame)
                    if len(buffer) == self.batch_size:
                        yield {"key": key, "start": count, "frames": torch.from_numpy(np.stack(buffer))}
                        count += len(buffer)
                        buffer = []
                if buffer:
                    yield {"key": key, "start": count, "frames": torch.from_numpy(np.stack(buffer))}
                    count += len(buffer)
                yield {"key": key, "end": count}
            except Exception as exc:
                yield {"key": key, "error": f"{type(exc).__name__}: {exc}"}


def _write_png(path: Path, labels: np.ndarray) -> None:
    if not cv2.imwrite(str(path), labels, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"Failed to write {path}")


def pending_sequences(root: Path, tag: str) -> tuple[list[tuple[str, dict]], int, int]:
    """Finished but unlabeled sequences (val first), plus the number of labeled and of all segments."""
    manifest = json.loads((root / "manifest.json").read_text())
    pending: list[tuple[str, bool, dict]] = []
    labeled = total = 0
    for video in manifest["videos"]:
        for segment in video["segments"]:
            total += 1
            directory = root / "sequences" / segment["sequence"]
            if (labels_dir(directory, tag) / "meta.json").exists():
                labeled += 1
            elif (directory / "meta.json").exists():
                meta = json.loads((directory / "meta.json").read_text())
                pending.append((segment["sequence"], video["split"] != "val", meta))
    pending.sort(key=lambda item: item[1])
    return [(name, meta) for name, _, meta in pending], labeled, total


def segment_sequences(
    root: Path,
    jobs: list[tuple[str, dict]],
    segmenter: Segmenter,
    tag: str,
    input_size: tuple[int, int],
    batch_size: int,
    workers: int,
    ffmpeg_threads: int,
    model_info: dict,
) -> list[str]:
    """Label ``jobs`` (``(sequence, frames meta.json)``); returns the names of failed sequences.

    ``model_info`` is copied into every labels ``meta.json``.
    """
    device = segmenter.mean.device
    in_h, in_w = input_size
    metas = dict(jobs)
    tmp_dirs: dict[str, Path] = {}
    for name in metas:
        final = labels_dir(root / "sequences" / name, tag)
        tmp = final.with_name(f".{final.name}.tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        tmp_dirs[name] = tmp

    loader = DataLoader(
        _FrameBatches(jobs, in_w, in_h, batch_size, ffmpeg_threads),
        batch_size=None,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=2 if workers > 0 else None,
    )
    failed: list[str] = []
    writes: dict[str, list[Future]] = {name: [] for name in metas}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for item in loader:
            name = item["key"]
            meta, tmp = metas[name], tmp_dirs[name]
            try:
                if "error" in item:
                    raise RuntimeError(item["error"])
                if "frames" in item:
                    out_size = (int(meta["height"]), int(meta["width"]))
                    labels = segmenter(item["frames"].to(device, non_blocking=True), out_size).cpu().numpy()
                    for k, label in enumerate(labels):
                        writes[name].append(pool.submit(_write_png, tmp / f"{item['start'] + k:06d}.png", label))
                    continue
                for future in writes.pop(name):
                    future.result()
                if item["end"] != int(meta["num_frames"]):
                    raise RuntimeError(f"decoded {item['end']} frames, the sequence has {meta['num_frames']}")
                label_meta = {
                    **model_info,
                    "sequence": name,
                    "num_frames": item["end"],
                    "width": int(meta["width"]),
                    "height": int(meta["height"]),
                    "input_size": [in_h, in_w],
                    "id2label": dict(enumerate(CORALSCAPES_CLASSES)),
                }
                (tmp / "meta.json").write_text(json.dumps(label_meta, indent=2))
                final = labels_dir(root / "sequences" / name, tag)
                if final.exists():
                    shutil.rmtree(final)
                tmp.rename(final)
                print(f"{name}: {item['end']} label maps", file=sys.stderr, flush=True)
            except Exception as exc:
                for future in writes.pop(name, []):
                    future.cancel()
                shutil.rmtree(tmp, ignore_errors=True)
                failed.append(name)
                print(f"FAILED {name}: {exc}", file=sys.stderr, flush=True)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="prepare_frames output (manifest.json + sequences/)")
    parser.add_argument("--tag", default=TAG, help="Labels go to sequences/<seq>/labels_<tag>/")
    parser.add_argument("--hf-repo", default=HF_REPO)
    parser.add_argument("--revision", default=HF_REVISION)
    parser.add_argument("--input-width", type=int, default=1024)
    parser.add_argument("--input-height", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8, help="Decoding processes (one sequence each)")
    parser.add_argument("--ffmpeg-threads", type=int, default=2, help="Decoder threads per worker")
    parser.add_argument("--no-amp", action="store_true", help="Run the segmenter in fp32 instead of bf16")
    parser.add_argument("--follow", action="store_true", help="Keep polling until every manifest segment is labeled")
    parser.add_argument("--poll-s", type=float, default=60.0)
    parser.add_argument(
        "--idle-timeout-s",
        type=float,
        default=6 * 3600.0,
        help="With --follow, stop after this long without a newly finished sequence",
    )
    args = parser.parse_args()
    if args.input_width % 16 or args.input_height % 16:
        raise SystemExit("--input-width and --input-height must be divisible by 16 (ViT patch size)")

    root = args.root.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segmenter = Segmenter.from_hub(args.hf_repo, args.revision, amp=not args.no_amp).to(device)
    model_info = {"hf_repo": args.hf_repo, "revision": args.revision}
    failed: set[str] = set()
    last_progress = time.monotonic()
    while True:
        jobs, labeled, total = pending_sequences(root, args.tag)
        jobs = [(name, meta) for name, meta in jobs if name not in failed]
        print(f"{labeled}/{total} segments labeled, {len(jobs)} ready, {len(failed)} failed", file=sys.stderr, flush=True)
        if jobs:
            failed.update(
                segment_sequences(
                    root,
                    jobs,
                    segmenter,
                    args.tag,
                    (args.input_height, args.input_width),
                    args.batch_size,
                    args.workers,
                    args.ffmpeg_threads,
                    model_info,
                )
            )
            last_progress = time.monotonic()
            continue
        if not args.follow or labeled + len(failed) >= total:
            break
        if time.monotonic() - last_progress > args.idle_timeout_s:
            print(f"No newly finished sequence for {args.idle_timeout_s:.0f} s; stopping", file=sys.stderr)
            break
        time.sleep(args.poll_s)
    if failed:
        raise SystemExit(f"{len(failed)} sequence(s) failed: {sorted(failed)}")


if __name__ == "__main__":
    main()
