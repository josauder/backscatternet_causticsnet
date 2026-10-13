"""Batched GPU augmentations with explicit per-sample parameters.

Color jitter follows ``torchvision.transforms.ColorJitter`` semantics (factors drawn from
``[1-x, 1+x]``, hue shift from ``[-h, h]``, the four operations applied in a random order
per sample) but lets all views of a clip share the same parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _grayscale(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x.unbind(dim=-3)
    return (0.2989 * r + 0.587 * g + 0.114 * b).unsqueeze(-3)


def rgb_to_hsv(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x.unbind(dim=-3)
    maxc = x.amax(dim=-3)
    minc = x.amin(dim=-3)
    eqc = maxc == minc
    cr = maxc - minc
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    cr_divisor = torch.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor
    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = torch.fmod((hr + hg + hb) / 6.0 + 1.0, 1.0)
    return torch.stack((h, s, maxc), dim=-3)


def hsv_to_rgb(x: torch.Tensor) -> torch.Tensor:
    h, s, v = x.unbind(dim=-3)
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    i = i.to(torch.int64) % 6
    p = (v * (1.0 - s)).clamp(0.0, 1.0)
    q = (v * (1.0 - s * f)).clamp(0.0, 1.0)
    t = (v * (1.0 - s * (1.0 - f))).clamp(0.0, 1.0)
    mask = (i.unsqueeze(-3) == torch.arange(6, device=x.device).view(-1, 1, 1)).to(x.dtype)
    a1 = torch.stack((v, q, p, p, t, v), dim=-3)
    a2 = torch.stack((t, v, v, q, p, p), dim=-3)
    a3 = torch.stack((p, p, t, v, v, q), dim=-3)
    a4 = torch.stack((a1, a2, a3), dim=-4)
    return torch.einsum("...ijk,...xijk->...xjk", mask, a4)


def adjust_brightness(x: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    return (x * factor).clamp(0.0, 1.0)


def adjust_contrast(x: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    mean = _grayscale(x).mean(dim=(-3, -2, -1), keepdim=True)
    return (factor * x + (1.0 - factor) * mean).clamp(0.0, 1.0)


def adjust_saturation(x: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    return (factor * x + (1.0 - factor) * _grayscale(x)).clamp(0.0, 1.0)


def adjust_hue(x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    hsv = rgb_to_hsv(x)
    h, s, v = hsv.unbind(dim=-3)
    h = torch.remainder(h + shift.squeeze(-3), 1.0)
    return hsv_to_rgb(torch.stack((h, s, v), dim=-3))


_OPS = (adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue)


@dataclass(frozen=True)
class ColorJitter:
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    hue: float = 0.0

    def sample(self, n: int, device: torch.device) -> dict[str, torch.Tensor]:
        def factor(x: float) -> torch.Tensor:
            return 1.0 + (torch.rand(n, device=device) * 2.0 - 1.0) * x

        return {
            "params": torch.stack(
                [
                    factor(self.brightness),
                    factor(self.contrast),
                    factor(self.saturation),
                    (torch.rand(n, device=device) * 2.0 - 1.0) * self.hue,
                ],
                dim=1,
            ),
            "order": torch.argsort(torch.rand(n, 4, device=device), dim=1),
        }

    def apply(self, images: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """``images [N, G, 3, H, W]`` in [0, 1]; all G images of sample n share its parameters."""
        out = images.clone()
        values = params["params"]
        order = params["order"]
        neutral = torch.tensor([1.0, 1.0, 1.0, 0.0], device=values.device)
        for position in range(4):
            for op_index, op in enumerate(_OPS):
                rows = (order[:, position] == op_index) & (values[:, op_index] != neutral[op_index])
                if rows.any():
                    value = values[rows, op_index].view(-1, 1, 1, 1, 1)
                    out[rows] = op(out[rows], value)
        return out

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.apply(images, self.sample(images.shape[0], images.device))


class Augmenter:
    """Pair-shared jitter, optional per-image jitter, and horizontal flips."""

    def __init__(
        self,
        jitter: dict | None = None,
        per_image_jitter: dict | None = None,
        hflip_p: float = 0.5,
    ) -> None:
        self.jitter = ColorJitter(**jitter) if jitter else None
        self.per_image = ColorJitter(**per_image_jitter) if per_image_jitter else None
        self.hflip_p = hflip_p

    def pair_jitter(self, images: torch.Tensor) -> torch.Tensor:
        """Same jitter for all views of a clip: ``images [B, V, 3, H, W]``."""
        return self.jitter(images) if self.jitter else images

    def per_image_jitter(self, images: torch.Tensor) -> torch.Tensor:
        """Independent jitter for every image of ``images [B, V, 3, H, W]``."""
        if self.per_image is None:
            raise RuntimeError("augment.per_image_jitter is not configured")
        return self.per_image(images.flatten(0, 1).unsqueeze(1)).squeeze(1).unflatten(0, images.shape[:2])

    def sample_flip(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch_size, device=device) < self.hflip_p


def hflip_images(images: torch.Tensor, flip: torch.Tensor) -> torch.Tensor:
    """Flip ``images [B, ...,  W]`` horizontally where ``flip [B]`` is true."""
    mask = flip.view(-1, *([1] * (images.dim() - 1)))
    return torch.where(mask, images.flip(-1), images)
