"""Frozen DeepReefMap SC-SfMLearner as geometry provider."""

from __future__ import annotations

from pathlib import Path

import torch

from caustics_backscatter.geometry.base import Geometry, GeometryProvider
from caustics_backscatter.geometry.pose import compose_chain, relative_poses
from caustics_backscatter.registry import GEOMETRY_PROVIDERS

_TRAIN_MEAN = 0.45
_TRAIN_STD = 0.225


@GEOMETRY_PROVIDERS.register("scsfmlearner")
class SCSfMLearnerGeometry(GeometryProvider):
    """Depth from ``DispResNet`` and hop poses from ``PoseResNet``, chained between views.

    Pose convention (from deepreefmap training): ``pose_net(a, b)`` is ``T_{b<-a}``.
    Chain frames are sorted in time, so each hop is ``pose_net(frame_k, frame_{k+1})``.

    ``input_norm="train"`` applies the ``(x - 0.45) / 0.225`` normalization used when the
    checkpoint was trained; ``"none"`` feeds ``[0, 1]`` images like deepreefmap's
    inference backend.
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        hf_repo: str = "EPFL-ECEO/deepreefmap-sfm-net",
        hf_filename: str = "scsfmlearner.pt",
        input_norm: str = "train",
        max_hop: int | None = 3,
        resnet_layers: int = 18,
        pose_chunk: int = 256,
    ) -> None:
        super().__init__()
        from deepreefmap.mapping.scsfmlearner.models import DispResNet, PoseResNet

        if input_norm not in {"train", "none"}:
            raise ValueError(f"input_norm must be 'train' or 'none', got {input_norm!r}")
        self.input_norm = input_norm
        self.max_hop = max_hop
        self.pose_chunk = pose_chunk

        path = Path(checkpoint) if checkpoint else self._download(hf_repo, hf_filename)
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.disp_net = DispResNet(num_layers=resnet_layers, pretrained=False)
        self.pose_net = PoseResNet(num_layers=18, pretrained=False)
        self.disp_net.load_state_dict(state["disp_state_dict"], strict=True)
        self.pose_net.load_state_dict(state["pose_state_dict"], strict=True)
        self.requires_grad_(False)
        self.eval()

    @staticmethod
    def _download(repo: str, filename: str) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=repo, filename=filename))

    def _normalize(self, frames_uint8: torch.Tensor) -> torch.Tensor:
        x = frames_uint8.float() / 255.0
        if self.input_norm == "train":
            x = (x - _TRAIN_MEAN) / _TRAIN_STD
        return x

    def depth(self, images: torch.Tensor) -> torch.Tensor:
        """``[N, 3, H, W]`` uint8 -> ``[N, 1, H, W]`` depth."""
        disp = self.disp_net(self._normalize(images))
        if isinstance(disp, list):
            disp = disp[0]
        return 1.0 / disp.clamp(min=1e-6)

    def pose(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """``T_{dst<-src}`` ``[N, 4, 4]`` for uint8 image batches ``[N, 3, H, W]``."""
        from deepreefmap.mapping.scsfmlearner.models import pose_vec_to_matrix

        out = []
        for start in range(0, src.shape[0], self.pose_chunk):
            a = self._normalize(src[start : start + self.pose_chunk])
            b = self._normalize(dst[start : start + self.pose_chunk])
            out.append(pose_vec_to_matrix(self.pose_net(a, b)))
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def forward(self, batch: dict[str, torch.Tensor]) -> Geometry:
        frames = batch["frames"]
        chain_valid = batch["chain_valid"]
        view_pos = batch["view_pos"]
        bsz, num_chain = frames.shape[:2]
        device = frames.device

        with torch.autocast(device_type=device.type, enabled=False):
            hops = torch.eye(4, device=device).expand(bsz, max(num_chain - 1, 0), 4, 4).clone()
            hop_valid = chain_valid[:, 1:]
            if num_chain > 1 and hop_valid.any():
                src = frames[:, :-1][hop_valid]
                dst = frames[:, 1:][hop_valid]
                hops[hop_valid] = self.pose(src, dst)
            transforms = relative_poses(compose_chain(hops), view_pos)

            batch_idx = torch.arange(bsz, device=device)[:, None]
            views = frames[batch_idx, view_pos]
            num_views = views.shape[1]
            depth = self.depth(views.flatten(0, 1)).unflatten(0, (bsz, num_views))

        return Geometry(depth=depth, T=transforms, K=batch["K"].float())
