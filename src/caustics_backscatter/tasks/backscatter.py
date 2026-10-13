from __future__ import annotations

from typing import Any

import numpy as np
import torch

from caustics_backscatter.losses.backscatternet import backscatternet_loss
from caustics_backscatter.models.backscatternet import backscatter
from caustics_backscatter.registry import TASKS
from caustics_backscatter.tasks.base import PreparedBatch, Task
from caustics_backscatter.utils.viz import depth_to_uint8, mask_to_uint8, to_uint8


@TASKS.register("backscatternet")
class BackscatterTask(Task):
    def predict_params(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, N, 3, H, W]`` -> ``(b_inf, beta)`` each ``[B, N, 3]``."""
        with self.autocast():
            b_inf, beta = self.model(images)
        return b_inf.float(), beta.float()

    def compute(self, prepared: PreparedBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        images = prepared.images
        depth = prepared.geometry.depth
        num_views = images.shape[1]
        use_style = self.loss_cfg["style"] > 0
        model_input = images
        if use_style:
            # One forward pass over the views and their per-image jittered copies.
            jittered = self.augmenter.per_image_jitter(images[:, self.inputs])
            model_input = torch.cat([images, jittered], dim=1)
        b_inf, beta = self.predict_params(model_input)

        scatter = backscatter(b_inf[:, :num_views], beta[:, :num_views], depth)
        descattered = images - scatter
        descattered_jittered = None
        if use_style:
            scatter_j = backscatter(b_inf[:, num_views:], beta[:, num_views:], depth[:, self.inputs])
            descattered_jittered = jittered - scatter_j

        total, logs = backscatternet_loss(
            descattered,
            scatter,
            images,
            prepared.geometry,
            prepared.weights,
            self.inputs,
            self.targets,
            descattered_jittered=descattered_jittered,
            ssim_weight=self.loss_cfg["ssim_weight"],
            style=self.loss_cfg["style"],
            backscatter_weight=self.loss_cfg["backscatter"],
            clip=self.loss_cfg["clip"],
            far_quantile=self.loss_cfg["far_quantile"],
            **self.warp_kwargs,
        )
        b_mean = b_inf[:, :num_views].detach().flatten(0, 1).mean(0)
        beta_mean = beta[:, :num_views].detach().flatten(0, 1).mean(0)
        for c, name in enumerate("rgb"):
            logs[f"b_inf_{name}"] = b_mean[c]
            logs[f"beta_{name}"] = beta_mean[c]
        logs["depth_median"] = depth.detach().median()
        logs["weight_mean"] = prepared.weights.mean()
        return total, logs, {"descattered": descattered, "backscatter": scatter}

    def visualize(self, prepared: PreparedBatch, outputs: dict[str, Any], num: int) -> dict[str, list[np.ndarray]]:
        view = self.inputs[0]
        rgb = prepared.images[:num, view]
        descattered = outputs["descattered"][:num, view]
        scatter = outputs["backscatter"][:num, view]
        # For display, add back the channel-mean of the removed backscatter to keep brightness.
        brightness_kept = descattered + scatter.mean(dim=1, keepdim=True)
        return {
            "rgb": [to_uint8(x) for x in rgb],
            "cleaned": [to_uint8(x) for x in descattered],
            "cleaned_brightness": [to_uint8(x) for x in brightness_kept],
            "backscatter": [to_uint8(x) for x in scatter],
            "depth": [depth_to_uint8(x) for x in prepared.geometry.depth[:num, view]],
            "weights": [mask_to_uint8(w[0]) for w in prepared.weights[:num, view]],
        }
