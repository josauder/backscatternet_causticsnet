"""Clips of undistorted video frames, including intermediate frames for pose chaining."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_jpeg, decode_png, read_file

from caustics_backscatter.data.sampler import ClipSampler


def labels_dir(directory: Path, tag: str) -> Path:
    """Per-frame class labels of a sequence written by ``segment_frames`` with ``--tag``."""
    return directory / f"labels_{tag}"


@dataclass(frozen=True)
class Sequence:
    name: str
    directory: Path
    num_frames: int
    intrinsics: np.ndarray  # [3, 3]
    index: int  # position among all segments of the manifest, stable while extraction runs

    def frame_path(self, index: int) -> Path:
        return self.directory / f"{index:06d}.jpg"

    def label_path(self, index: int, tag: str) -> Path:
        return labels_dir(self.directory, tag) / f"{index:06d}.png"


def load_sequences(root: str | Path, split: str, labels: str | None = None) -> list[Sequence]:
    """Sequences of ``split`` that ``prepare_frames`` has finished (``meta.json`` exists).

    With ``labels``, only sequences whose ``labels_<labels>/`` ``segment_frames`` has
    finished are returned. Unfinished segments are skipped, so calling this again later
    picks up the sequences finished in the meantime. May return an empty list.
    """
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    sequences = []
    index = 0
    for video in manifest["videos"]:
        for segment in video["segments"]:
            directory = root / "sequences" / segment["sequence"]
            meta_path = directory / "meta.json"
            labeled = labels is None or (labels_dir(directory, labels) / "meta.json").exists()
            if video["split"] == split and meta_path.exists() and labeled:
                meta = json.loads(meta_path.read_text())
                intrinsics = np.loadtxt(directory / "cam.txt", dtype=np.float32).reshape(3, 3)
                sequences.append(Sequence(segment["sequence"], directory, int(meta["num_frames"]), intrinsics, index))
            index += 1
    return sequences


class ClipDataset(Dataset):
    """Each item is one clip: the chain frames needed to compute geometry between its views.

    Items (padded to ``sampler.max_chain`` so batches stack):
        frames ``[C, 3, H, W]`` uint8, chain_valid ``[C]`` bool, frame_ids ``[C]`` (-1 = pad),
        view_pos ``[V]`` chain position of each view, seq_idx, K ``[3, 3]``, and with
        ``labels`` (a ``segment_frames`` tag) the class ids of the views ``[V, 1, H, W]`` uint8.

    Training (``deterministic=False``) ignores the index and draws a random sequence
    (weighted by length), anchor and gaps from the worker's torch RNG. Validation
    (``deterministic=True``) derives the clip from ``seed + index`` so it is fixed across
    evals for a fixed set of sequences.

    ``set_sequences`` swaps in a new set of sequences (e.g. after more were extracted);
    DataLoader workers see it from the next ``iter(loader)`` unless they are persistent.
    """

    def __init__(
        self,
        sequences: list[Sequence],
        sampler: ClipSampler,
        image_size: tuple[int, int],
        length: int,
        deterministic: bool = False,
        seed: int = 0,
        labels: str | None = None,
    ) -> None:
        self.sampler = sampler
        self.image_size = tuple(image_size)
        self.length = length
        self.deterministic = deterministic
        self.seed = seed
        self.labels = labels
        self.set_sequences(sequences)

    def set_sequences(self, sequences: list[Sequence]) -> None:
        """Use the sequences long enough for the clip span (possibly none)."""
        max_span = self.sampler.max_span
        self.sequences = [s for s in sequences if s.num_frames > max_span]
        lengths = np.array([s.num_frames - max_span for s in self.sequences], dtype=np.float64)
        self.sequence_probs = lengths / lengths.sum() if len(lengths) else lengths

    @property
    def num_frames(self) -> int:
        return sum(s.num_frames for s in self.sequences)

    def __len__(self) -> int:
        return self.length

    def _rng(self, index: int) -> np.random.Generator:
        if self.deterministic:
            return np.random.default_rng([self.seed, index])
        return np.random.default_rng(int(torch.randint(0, 2**62, (1,)).item()))

    def _read(self, path: Path) -> torch.Tensor:
        image = decode_jpeg(read_file(str(path)), mode=ImageReadMode.RGB)
        if tuple(image.shape[-2:]) != self.image_size:
            raise ValueError(f"{path} has size {tuple(image.shape[-2:])}, expected {self.image_size}")
        return image

    def _read_labels(self, path: Path) -> torch.Tensor:
        labels = decode_png(read_file(str(path)), mode=ImageReadMode.UNCHANGED)
        if tuple(labels.shape) != (1, *self.image_size):
            raise ValueError(f"{path} has shape {tuple(labels.shape)}, expected {(1, *self.image_size)}")
        return labels

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not self.sequences:
            raise RuntimeError("No sequence is long enough for the requested clip span")
        rng = self._rng(index)
        sequence = self.sequences[int(rng.choice(len(self.sequences), p=self.sequence_probs))]
        offsets = self.sampler.offsets(rng)
        anchor = int(rng.integers(0, sequence.num_frames - max(offsets)))
        chain, view_pos = self.sampler.chain([anchor + o for o in offsets])

        max_chain = self.sampler.max_chain
        frames = [self._read(sequence.frame_path(f)) for f in chain]
        frames += [frames[-1]] * (max_chain - len(chain))
        chain_valid = torch.zeros(max_chain, dtype=torch.bool)
        chain_valid[: len(chain)] = True
        frame_ids = torch.full((max_chain,), -1, dtype=torch.long)
        frame_ids[: len(chain)] = torch.tensor(chain)
        item = {
            "frames": torch.stack(frames),
            "chain_valid": chain_valid,
            "frame_ids": frame_ids,
            "view_pos": torch.tensor(view_pos, dtype=torch.long),
            "seq_idx": torch.tensor(sequence.index, dtype=torch.long),
            "K": torch.from_numpy(sequence.intrinsics.copy()),
        }
        if self.labels is not None:
            item["labels"] = torch.stack([self._read_labels(sequence.label_path(chain[p], self.labels)) for p in view_pos])
        return item
