"""A task bundles the trainable model, the frozen geometry, augmentations and the objective."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from caustics_backscatter.data.augment import Augmenter, hflip_images
from caustics_backscatter.geometry.base import Geometry
from caustics_backscatter.models.bn import set_bn_momentum
from caustics_backscatter.registry import EVALUATORS, GEOMETRY_PROVIDERS, MODELS, PIXEL_WEIGHTERS


@dataclass
class PreparedBatch:
    """Views of a batch of clips after geometry estimation and augmentation.

    raw: ``[B, V, 3, H, W]`` un-jittered views in [0, 1] (already flipped where ``flip``).
    images: ``raw`` with the clip-shared color jitter (identical to ``raw`` without augmentation).
    geometry: depth/poses/intrinsics consistent with the (flipped) views.
    weights: ``[B, V, 1, H, W]`` pixel weights from the weighter.
    """

    raw: torch.Tensor
    images: torch.Tensor
    geometry: Geometry
    weights: torch.Tensor
    flip: torch.Tensor


class Task(ABC):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.model = MODELS.build(cfg.model)
        set_bn_momentum(self.model, cfg.train.bn_momentum)
        self.geometry = GEOMETRY_PROVIDERS.build(cfg.geometry)
        self.weighter = PIXEL_WEIGHTERS.build(cfg.weighting)
        self.augmenter = Augmenter(**OmegaConf.to_container(cfg.augment, resolve=True))
        self.inputs = list(cfg.views.inputs)
        self.targets = list(cfg.views.targets)
        self.warp_kwargs = dict(OmegaConf.to_container(cfg.warp, resolve=True))
        self.loss_cfg = OmegaConf.to_container(cfg.loss, resolve=True)
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[cfg.train.amp]
        self.device = torch.device("cpu")

    def to(self, device: torch.device) -> Task:
        self.device = device
        self.model.to(device)
        if self.cfg.train.channels_last:
            self.model.to(memory_format=torch.channels_last)
        self.geometry.to(device)
        self.weighter.to(device)
        return self

    def autocast(self) -> AbstractContextManager:
        return torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype or torch.float32, enabled=self.amp_dtype is not None
        )

    def prepare(self, batch: dict[str, torch.Tensor], augment: bool) -> PreparedBatch:
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        geometry = self.geometry(batch)
        frames, view_pos = batch["frames"], batch["view_pos"]
        bsz = frames.shape[0]
        views = frames[torch.arange(bsz, device=self.device)[:, None], view_pos].float() / 255.0
        labels = batch.get("labels")
        if augment and self.augmenter.hflip_p > 0:
            flip = self.augmenter.sample_flip(bsz, self.device)
            views = hflip_images(views, flip)
            if labels is not None:
                labels = hflip_images(labels, flip)
            geometry = geometry.hflip(flip)
        else:
            flip = torch.zeros(bsz, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            weights = self.weighter(views, labels)
        images = self.augmenter.pair_jitter(views) if augment else views
        return PreparedBatch(raw=views, images=images, geometry=geometry, weights=weights, flip=flip)

    def training_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total, logs, _ = self.compute(self.prepare(batch, augment=True))
        return total, logs

    @abstractmethod
    def compute(
        self, prepared: PreparedBatch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        """Return ``(total_loss, scalar_logs, outputs_for_visualization)``."""

    @abstractmethod
    def visualize(self, prepared: PreparedBatch, outputs: dict[str, Any], num: int) -> dict[str, list[np.ndarray]]:
        """W&B image panels: panel name -> list of ``[H, W, 3]`` uint8 images."""

    def build_evaluators(self) -> list:
        return [EVALUATORS.build(ev) for ev in self.cfg.eval.evaluators]
