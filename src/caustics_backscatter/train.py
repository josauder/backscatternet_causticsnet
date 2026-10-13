"""Train CausticsNet or BackscatterNet.

    python -m caustics_backscatter.train --config configs/causticsnet.yaml [key=value ...]

Checkpoints go to ``<paths.runs_root>/<run_name>/checkpoints``; an existing ``latest.pt``
is resumed automatically (including the W&B run), so jobs can be requeued.

Training can start while ``prepare_frames`` / ``segment_frames`` are still running: it
waits until at least one train and one val sequence are finished (and labeled, with
``data.labels``), and rescans the dataset at every epoch (``train.epoch_steps`` steps) and
before every eval, so newly extracted sequences are used.
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import random
import sys
import time
import traceback
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from caustics_backscatter.config import load_config
from caustics_backscatter.data.dataset import ClipDataset, load_sequences
from caustics_backscatter.data.sampler import ClipSampler
from caustics_backscatter.registry import TASKS
from caustics_backscatter.tasks import Task


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_sampler(cfg: DictConfig) -> ClipSampler:
    return ClipSampler(
        num_views=cfg.views.num,
        gap_min=cfg.views.gap_min,
        gap_max=cfg.views.gap_max,
        max_hop=cfg.geometry.get("max_hop"),
        shuffle=cfg.views.shuffle,
    )


def build_loaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    sampler = build_sampler(cfg)
    image_size = tuple(cfg.data.image_size)
    labels = cfg.data.get("labels")
    train_ds = ClipDataset(
        load_sequences(cfg.data.root, cfg.data.train_split, labels),
        sampler,
        image_size,
        length=cfg.train.epoch_steps * cfg.train.batch_size,
        labels=labels,
    )
    val_ds = ClipDataset(
        load_sequences(cfg.data.root, cfg.data.val_split, labels),
        sampler,
        image_size,
        length=cfg.data.num_val_clips,
        deterministic=True,
        seed=cfg.seed,
        labels=labels,
    )
    workers = cfg.train.num_workers
    # Workers must not be persistent: each epoch they are re-forked so they see the
    # sequences added by refresh_dataset.
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
        prefetch_factor=4 if workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.eval.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def refresh_dataset(loader: DataLoader, root: str, split: str) -> dict[str, int]:
    """Rescan ``root`` for finished (and, if the dataset uses labels, labeled) sequences of
    ``split``; takes effect at the next ``iter(loader)``."""
    dataset: ClipDataset = loader.dataset
    dataset.set_sequences(load_sequences(root, split, dataset.labels))
    return {f"data/num_{split}_sequences": len(dataset.sequences), f"data/num_{split}_frames": dataset.num_frames}


def wait_for_data(cfg: DictConfig, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, int]:
    while True:
        stats = {
            **refresh_dataset(train_loader, cfg.data.root, cfg.data.train_split),
            **refresh_dataset(val_loader, cfg.data.root, cfg.data.val_split),
        }
        if train_loader.dataset.sequences and val_loader.dataset.sequences:
            return stats
        print(f"Waiting for usable sequences under {cfg.data.root} ({stats}); retrying in {cfg.data.poll_s}s", flush=True)
        time.sleep(cfg.data.poll_s)


class Checkpointer:
    def __init__(self, directory: Path, keep_steps: bool) -> None:
        self.directory = directory
        self.keep_steps = keep_steps
        directory.mkdir(parents=True, exist_ok=True)

    @property
    def latest(self) -> Path:
        return self.directory / "latest.pt"

    def save(self, task: Task, optimizer: torch.optim.Optimizer, step: int, cfg: DictConfig) -> None:
        state = {
            "step": step,
            "model": task.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        tmp = self.latest.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, self.latest)
        if self.keep_steps:
            torch.save(
                {"step": step, "model": state["model"], "config": state["config"]},
                self.directory / f"step_{step:06d}.pt",
            )

    def resume(self, task: Task, optimizer: torch.optim.Optimizer) -> int:
        if not self.latest.exists():
            return 0
        state = torch.load(self.latest, map_location="cpu", weights_only=False)
        task.model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        print(f"Resumed from {self.latest} at step {state['step']}")
        return int(state["step"])


class WandbLogger:
    def __init__(self, cfg: DictConfig, out_dir: Path) -> None:
        import wandb

        self.wandb = wandb
        id_file = out_dir / "wandb_run_id.txt"
        run_id = id_file.read_text().strip() if id_file.exists() else uuid.uuid4().hex[:12]
        id_file.write_text(run_id)
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            name=cfg.run_name,
            id=run_id,
            resume="allow",
            mode=cfg.wandb.mode,
            dir=str(out_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    def log(self, metrics: dict, step: int, commit: bool | None = None) -> None:
        self.wandb.log(metrics, step=step, commit=commit)

    def use_eval_axis(self, prefixes: list[str]) -> None:
        """Plot ``<prefix>/*`` against ``eval_step`` (the training step they evaluate)."""
        self.wandb.define_metric("eval_step")
        for prefix in prefixes:
            self.wandb.define_metric(f"{prefix}/*", step_metric="eval_step")

    def log_eval(self, metrics: dict, step: int) -> None:
        """Log on a new, immediately uploaded row, tagged with ``eval_step = step``."""
        self.wandb.log({**metrics, "eval_step": step})

    def images(self, prefix: str, panels: dict[str, list[np.ndarray]]) -> dict:
        return {
            f"{prefix}/{name}": [self.wandb.Image(img, caption=f"{name} {i}") for i, img in enumerate(images)]
            for name, images in panels.items()
        }

    def finish(self) -> None:
        self.wandb.finish()


def _to_float(logs: dict[str, torch.Tensor]) -> dict[str, float]:
    return {k: float(v) for k, v in logs.items()}


def _eval_log(message: str) -> None:
    print(f"[eval] {message}", file=sys.stderr, flush=True)


@torch.no_grad()
def evaluate(
    task: Task,
    val_loader: DataLoader,
    train_batch: dict[str, torch.Tensor],
    evaluators: list,
    logger: WandbLogger,
    step: int,
    extra: dict,
) -> None:
    """Val losses and train/val panels are uploaded before the (possibly slow) evaluators run."""
    task.model.eval()
    start = time.perf_counter()
    num_batches = len(val_loader)
    _eval_log(f"step {step}: validation on {len(val_loader.dataset)} clips ({num_batches} batches)")
    num_viz = task.cfg.eval.num_viz
    sums: dict[str, float] = defaultdict(float)
    batches = 0
    metrics: dict = {}
    for k, batch in enumerate(val_loader):
        if k == 0:
            _eval_log(f"step {step}: first val batch after {time.perf_counter() - start:.1f} s (loader startup)")
        prepared = task.prepare(batch, augment=False)
        _, logs, outputs = task.compute(prepared)
        for name, value in _to_float(logs).items():
            sums[name] += value
        batches += 1
        if k == 0:
            metrics.update(logger.images("val", task.visualize(prepared, outputs, num_viz)))
        _eval_log(f"step {step}: val batch {batches}/{num_batches} ({time.perf_counter() - start:.1f} s)")
    metrics.update({f"val/{name}": value / max(batches, 1) for name, value in sums.items()})

    prepared = task.prepare(train_batch, augment=True)
    _, _, outputs = task.compute(prepared)
    metrics.update(logger.images("train", task.visualize(prepared, outputs, num_viz)))
    metrics.update(extra)
    _eval_log(f"step {step}: train panels done ({time.perf_counter() - start:.1f} s); logging to W&B")
    # Without commit, W&B holds back a row logged with an explicit step until a later step is
    # logged, i.e. until after the evaluators.
    logger.log(metrics, step, commit=True)
    _eval_log(
        f"step {step}: val loss {metrics.get('val/loss', float('nan')):.5f}; "
        f"val/train panels logged to W&B ({time.perf_counter() - start:.0f} s)"
    )

    for evaluator in evaluators:
        start = time.perf_counter()
        _eval_log(f"step {step}: running {evaluator.name}")
        scalars, panels = evaluator(task)
        logger.log_eval({**scalars, **logger.images(evaluator.name, panels)}, step)
        _eval_log(f"step {step}: {evaluator.name} logged to W&B ({(time.perf_counter() - start) / 60:.1f} min)")
    task.model.train()


def train(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    out_dir = Path(cfg.paths.runs_root) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    task: Task = TASKS.build(cfg.task, cfg=cfg)
    task.to(device)
    train_loader, val_loader = build_loaders(cfg)
    optimizer = torch.optim.Adam(
        task.model.parameters(),
        lr=cfg.train.lr,
        betas=tuple(cfg.train.betas),
        weight_decay=cfg.train.weight_decay,
    )
    checkpointer = Checkpointer(out_dir / "checkpoints", keep_steps=cfg.train.keep_step_checkpoints)
    step = checkpointer.resume(task, optimizer)
    if step >= cfg.train.max_steps:
        print(f"Run already finished at step {step}")
        return
    data_stats = wait_for_data(cfg, train_loader, val_loader)
    print(f"Data: {data_stats}")
    logger = WandbLogger(cfg, out_dir)
    evaluators = task.build_evaluators()
    logger.use_eval_axis([evaluator.name for evaluator in evaluators])

    task.model.train()
    train_iter = iter(train_loader)
    sums: dict[str, torch.Tensor] = {}
    count = 0
    data_time = 0.0
    window_start = time.perf_counter()
    try:
        while step < cfg.train.max_steps:
            t0 = time.perf_counter()
            try:
                batch = next(train_iter)
            except StopIteration:
                data_stats.update(refresh_dataset(train_loader, cfg.data.root, cfg.data.train_split))
                print(f"New epoch at step {step}: {data_stats}")
                train_iter = iter(train_loader)
                batch = next(train_iter)
            data_time += time.perf_counter() - t0

            loss, logs = task.training_step(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.train.grad_clip:
                logs["grad_norm"] = torch.nn.utils.clip_grad_norm_(task.model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            step += 1

            for name, value in logs.items():
                sums[name] = sums.get(name, 0.0) + value.detach()
            count += 1
            if step % cfg.train.log_every == 0:
                elapsed = time.perf_counter() - window_start
                metrics = {f"train/{name}": float(value) / count for name, value in sums.items()}
                metrics["train/lr"] = optimizer.param_groups[0]["lr"]
                metrics["perf/clips_per_s"] = count * cfg.train.batch_size / elapsed
                metrics["perf/data_frac"] = data_time / elapsed
                metrics.update(data_stats)
                logger.log(metrics, step)
                print(f"step {step}: loss={metrics['train/loss']:.5f} ({metrics['perf/clips_per_s']:.1f} clips/s)")
                sums.clear()
                count = 0
                data_time = 0.0
                window_start = time.perf_counter()

            if step % cfg.train.eval_every == 0 or step == cfg.train.max_steps:
                data_stats.update(refresh_dataset(val_loader, cfg.data.root, cfg.data.val_split))
                evaluate(task, val_loader, batch, evaluators, logger, step, data_stats)
                checkpointer.save(task, optimizer, step, cfg)
                window_start = time.perf_counter()
    except BaseException:
        # Print now: logger.finish() waits for the W&B upload, which can take long or hang.
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        logger.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. train.batch_size=64")
    args = parser.parse_args()
    faulthandler.enable()  # Python stack on segfaults / fatal signals instead of a silent death
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
