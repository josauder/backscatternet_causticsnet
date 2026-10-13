from __future__ import annotations

from typing import Any

import numpy as np
import torch

from caustics_backscatter.losses.causticsnet import causticsnet_loss
from caustics_backscatter.registry import TASKS
from caustics_backscatter.tasks.base import PreparedBatch, Task
from caustics_backscatter.utils.viz import mask_to_uint8, signed_to_uint8, to_uint8


@TASKS.register("causticsnet")
class CausticsTask(Task):
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """``[B, N, 3, H, W]`` -> caustics ``[B, N, 3, H, W]`` (float32)."""
        with self.autocast():
            return self.model(images).float()

    def predict_caustics(self, images: torch.Tensor) -> torch.Tensor:
        """Single images ``[B, 3, H, W]`` -> caustics ``[B, 3, H, W]``."""
        return self.predict(images[:, None])[:, 0]

    def compute(self, prepared: PreparedBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        caustics = self.predict(prepared.images[:, self.inputs])
        total, logs, residuals = causticsnet_loss(
            caustics,
            prepared.images,
            prepared.geometry,
            prepared.weights,
            self.inputs,
            self.targets,
            tv=self.loss_cfg["tv"],
            l2_weight=self.loss_cfg["l2"],
            clip=self.loss_cfg["clip"],
            **self.warp_kwargs,
        )
        logs["weight_mean"] = prepared.weights.mean()
        return total, logs, {"caustics": caustics, "residuals": residuals}

    def visualize(self, prepared: PreparedBatch, outputs: dict[str, Any], num: int) -> dict[str, list[np.ndarray]]:
        gain = self.cfg.viz.residual_gain
        view = self.inputs[0]
        rgb = prepared.images[:num, view]
        caustics = outputs["caustics"][:num, 0]
        target = outputs["residuals"][view][:num]
        return {
            "rgb": [to_uint8(x) for x in rgb],
            "cleaned": [to_uint8(x) for x in rgb - caustics],
            "caustics": [signed_to_uint8(x, gain) for x in caustics],
            "target": [signed_to_uint8(x, gain) for x in target],
            "weights": [mask_to_uint8(w[0]) for w in prepared.weights[:num, view]],
        }
