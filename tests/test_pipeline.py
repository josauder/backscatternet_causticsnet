"""End-to-end CPU smoke tests on a tiny synthetic dataset with a randomly initialized SC-SfMLearner."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from caustics_backscatter.config import load_config
from caustics_backscatter.registry import TASKS
from caustics_backscatter.tasks import Task
from caustics_backscatter.train import build_loaders

H, W = 64, 128
TAG = "coralscapesv2_vitb39"  # data.labels in configs/base.yaml


def _write_sequence(
    directory: Path, num_frames: int, rng: np.random.Generator, meta: bool = True, labels: bool = False
) -> None:
    directory.mkdir(parents=True)
    base = rng.integers(0, 255, size=(H, W + num_frames, 3), dtype=np.uint8)
    for i in range(num_frames):
        cv2.imwrite(str(directory / f"{i:06d}.jpg"), base[:, i : i + W])
    k = np.array([[100.0, 0, (W - 1) / 2], [0, 100.0, (H - 1) / 2], [0, 0, 1]], dtype=np.float32)
    np.savetxt(directory / "cam.txt", k)
    if meta:
        _finish_sequence(directory, num_frames)
    if labels:
        _write_labels(directory, num_frames, rng)


def _finish_sequence(directory: Path, num_frames: int) -> None:
    (directory / "meta.json").write_text(json.dumps({"num_frames": num_frames}))


def _write_labels(directory: Path, num_frames: int, rng: np.random.Generator) -> None:
    from caustics_backscatter.data.dataset import labels_dir

    out = labels_dir(directory, TAG)
    out.mkdir()
    for i in range(num_frames):
        cv2.imwrite(str(out / f"{i:06d}.png"), rng.integers(0, 40, size=(H, W), dtype=np.uint8))
    (out / "meta.json").write_text(json.dumps({"num_frames": num_frames}))


def _write_manifest(root: Path, videos: dict[str, tuple[str, list[str]]]) -> None:
    entries = [
        {"sequence": name, "split": split, "segments": [{"sequence": s, "start_s": 0.0, "end_s": 3.0} for s in segments]}
        for name, (split, segments) in videos.items()
    ]
    (root / "manifest.json").write_text(json.dumps({"videos": entries, "camera_profile": "gopro_hero_10"}))


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    from deepreefmap.mapping.scsfmlearner.models import DispResNet, PoseResNet

    root = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    for name in ("seq_a__seg0", "seq_b__seg0"):
        _write_sequence(root / "sequences" / name, 60, rng, labels=True)
    _write_sequence(root / "sequences" / "seq_a__seg1", 60, rng)  # not segmented yet
    _write_manifest(root, {"seq_a": ("train", ["seq_a__seg0", "seq_a__seg1"]), "seq_b": ("val", ["seq_b__seg0"])})

    ckpt = root / "scsfm.pt"
    torch.save(
        {
            "disp_state_dict": DispResNet(num_layers=18, pretrained=False).state_dict(),
            "pose_state_dict": PoseResNet(num_layers=18, pretrained=False).state_dict(),
        },
        ckpt,
    )
    return root, ckpt


def _config(name: str, root: Path, ckpt: Path, extra=()):
    return load_config(
        f"configs/{name}.yaml",
        [
            f"data.root={root}",
            f"data.image_size=[{H},{W}]",
            "data.num_val_clips=4",
            f"geometry.checkpoint={ckpt}",
            "model.encoder_name=resnet18",
            "model.encoder_weights=null",
            "train.amp=none",
            "train.channels_last=false",
            "train.batch_size=2",
            "train.num_workers=0",
            "train.max_steps=2",
            "eval.batch_size=2",
            "eval.num_workers=0",
            "eval.evaluators=[]",
            *extra,
        ],
    )


@pytest.mark.parametrize("name", ["causticsnet", "backscatternet"])
def test_training_step_and_visualization(setup, name):
    root, ckpt = setup
    cfg = _config(name, root, ckpt)
    task: Task = TASKS.build(cfg.task, cfg=cfg)
    train_loader, val_loader = build_loaders(cfg)
    assert [s.name for s in train_loader.dataset.sequences] == ["seq_a__seg0"]
    batch = next(iter(train_loader))
    max_chain = 3 if name == "causticsnet" else 8
    assert batch["frames"].shape == (2, max_chain, 3, H, W)
    assert batch["labels"].shape == (2, 2, 1, H, W) and batch["labels"].dtype == torch.uint8

    prepared = task.prepare(batch, augment=True)
    labels = torch.where(prepared.flip.view(-1, 1, 1, 1, 1), batch["labels"].flip(-1), batch["labels"])
    zeroed = torch.isin(labels, torch.tensor([7, 8, 9, 15], dtype=torch.uint8))
    assert torch.equal(prepared.weights == 0, zeroed)

    task.model.train()
    loss, logs = task.training_step(batch)
    assert torch.isfinite(loss)
    assert 0 < float(logs["weight_mean"]) < 1
    loss.backward()
    grads = [p.grad for p in task.model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert all(p.grad is None for p in task.geometry.parameters())

    task.model.eval()
    with torch.no_grad():
        prepared = task.prepare(next(iter(val_loader)), augment=False)
        _, _, outputs = task.compute(prepared)
        panels = task.visualize(prepared, outputs, num=2)
    assert "rgb" in panels and "cleaned" in panels and "weights" in panels
    assert all(img.shape == (H, W, 3) and img.dtype == np.uint8 for imgs in panels.values() for img in imgs)


def test_uniform_weighting_without_labels(setup):
    root, ckpt = setup
    cfg = _config("causticsnet", root, ckpt, ["data.labels=null", "weighting.name=uniform"])
    task: Task = TASKS.build(cfg.task, cfg=cfg)
    train_loader, _ = build_loaders(cfg)
    assert [s.name for s in train_loader.dataset.sequences] == ["seq_a__seg0", "seq_a__seg1"]
    batch = next(iter(train_loader))
    assert "labels" not in batch
    assert torch.equal(task.prepare(batch, augment=True).weights, torch.ones(2, 2, 1, H, W))


def test_incomplete_dataset_is_refreshed(tmp_path):
    from caustics_backscatter.data.dataset import ClipDataset, load_sequences
    from caustics_backscatter.data.sampler import ClipSampler

    rng = np.random.default_rng(2)
    _write_manifest(
        tmp_path,
        {"v0": ("train", ["v0__seg0", "v0__seg1"]), "v1": ("val", ["v1__seg0"]), "v2": ("train", ["v2__seg0"])},
    )
    sampler = ClipSampler(num_views=2, gap_min=3, gap_max=6, max_hop=3)
    dataset = ClipDataset(load_sequences(tmp_path, "train"), sampler, (H, W), length=8)
    assert dataset.sequences == []
    with pytest.raises(RuntimeError):
        dataset[0]

    _write_sequence(tmp_path / "sequences" / "v2__seg0", 30, rng)
    _write_sequence(tmp_path / "sequences" / "v0__seg1", 30, rng, meta=False)  # still being extracted
    dataset.set_sequences(load_sequences(tmp_path, "train"))
    assert [s.name for s in dataset.sequences] == ["v2__seg0"]
    assert int(next(iter(DataLoader(dataset, batch_size=4)))["seq_idx"].unique()) == 3

    _finish_sequence(tmp_path / "sequences" / "v0__seg1", 30)
    dataset.set_sequences(load_sequences(tmp_path, "train"))
    assert [(s.name, s.index) for s in dataset.sequences] == [("v0__seg1", 1), ("v2__seg0", 3)]
    assert dataset.num_frames == 60
    assert load_sequences(tmp_path, "val") == []

    assert load_sequences(tmp_path, "train", TAG) == []
    _write_labels(tmp_path / "sequences" / "v2__seg0", 30, rng)
    assert [s.name for s in load_sequences(tmp_path, "train", TAG)] == ["v2__seg0"]


def test_rcaustics_evaluator(setup, tmp_path):
    from caustics_backscatter.eval.rcaustics import RCausticsEvaluator, find_rcaustics_samples

    root, ckpt = setup
    rng = np.random.default_rng(1)
    for split in (1, 2):
        folder = tmp_path / "with_caustics-free_ground truth" / f"dataset_{split}_L"
        folder.mkdir(parents=True)
        for i in range(2):
            stem = f"subset_{split}_L_{i:03d}"
            cv2.imwrite(str(folder / f"{stem}.JPG"), rng.integers(0, 255, size=(90, 120, 3), dtype=np.uint8))
            mask = (rng.random((90, 120)) < 0.2).astype(np.uint8) * 255
            cv2.imwrite(str(folder / f"{stem}.bin.JPG"), mask)
        (folder / f"subset_{split}_L.back.JPG").write_bytes(b"")
    assert len(find_rcaustics_samples(tmp_path)) == 4

    cfg = _config("causticsnet", root, ckpt)
    task: Task = TASKS.build(cfg.task, cfg=cfg)
    task.model.eval()
    evaluator = RCausticsEvaluator(str(tmp_path), input_size=(64, 96), num_bins=100, num_workers=0, num_viz=2, viz_width=60)
    scalars, panels = evaluator(task)
    assert 0 <= scalars["rcaustics/global/f1/1-7"] <= 100
    assert "rcaustics/per_split/acc/1,2" in scalars
    assert len(panels["pred_mask"]) == 2
