"""Backscatter is predicted on the RGB frame and removed from the caustics-cleaned frame."""

import math

import numpy as np
import pytest
import torch

from caustics_backscatter.infer import (
    PanelComposer,
    infer_batch,
    load_trained_model,
    make_rectifier,
    render_panels,
    resize_frame,
    run_video,
    title_bar_height,
)
from caustics_backscatter.models.backscatternet import backscatter


class _Caustics:
    def __init__(self, value: float) -> None:
        self.value = value
        self.seen = None

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        self.seen = images.detach().clone()
        return torch.full_like(images, self.value)


class _Backscatter:
    def __init__(self, b_inf: tuple[float, float, float], beta: float) -> None:
        self.b_inf = b_inf
        self.beta = beta
        self.seen = None

    def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.seen = images.detach().clone()
        b_inf = images.new_tensor(self.b_inf).view(1, 1, 3).expand(images.shape[0], images.shape[1], 3).contiguous()
        beta = torch.full_like(b_inf, self.beta)
        return b_inf, beta


class _Depth:
    def __init__(self, value: float = 1.0) -> None:
        self.value = value
        self.seen = None

    def depth(self, images: torch.Tensor) -> torch.Tensor:
        self.seen = images.detach().clone()
        return torch.full((images.shape[0], 1, images.shape[2], images.shape[3]), self.value, device=images.device)


def test_backscatter_is_estimated_on_rgb_and_removed_from_caustics_cleaned():
    rgb_u8 = torch.tensor([180, 40, 20], dtype=torch.uint8).view(1, 3, 1, 1).expand(2, 3, 8, 16).contiguous()
    caustics_net = _Caustics(0.25)
    backscatter_net = _Backscatter((0.2, 0.4, 0.6), beta=1.0)
    geometry = _Depth(1.0)

    outputs = infer_batch(rgb_u8, caustics_net, backscatter_net, geometry, keep_brightness=True)

    rgb = rgb_u8.float() / 255.0
    assert torch.equal(caustics_net.seen, rgb[:, None])
    assert torch.equal(backscatter_net.seen, rgb[:, None])
    assert torch.equal(geometry.seen, rgb_u8)

    cleaned = (rgb - 0.25).clamp(0, 1)
    b_inf = torch.tensor([0.2, 0.4, 0.6]).expand(2, 3)
    veil = backscatter(b_inf, torch.ones(2, 3), torch.ones(2, 1, 8, 16))
    assert torch.allclose(outputs["caustics_cleaned"], cleaned)
    assert torch.allclose(outputs["backscatter"], veil)
    kept = (cleaned - veil + veil.mean(dim=1, keepdim=True)).clamp(0, 1)
    assert torch.allclose(outputs["backscatter_cleaned"], kept)
    # The cleaned frame is the one the veil is removed from, so it differs from removing it from RGB.
    from_rgb = (rgb - veil + veil.mean(dim=1, keepdim=True)).clamp(0, 1)
    assert not torch.allclose(outputs["backscatter_cleaned"], from_rgb)


def test_brightness_compensation_can_be_turned_off():
    rgb_u8 = torch.full((1, 3, 4, 4), 200, dtype=torch.uint8)
    outputs = infer_batch(rgb_u8, _Caustics(0.1), _Backscatter((0.3, 0.3, 0.3), beta=1.0), _Depth(), keep_brightness=False)
    rgb = rgb_u8.float() / 255.0
    cleaned = (rgb - 0.1).clamp(0, 1)
    veil = outputs["backscatter"]
    assert torch.allclose(outputs["backscatter_cleaned"], (cleaned - veil).clamp(0, 1))


def test_caustics_panel_is_the_gray_channel_mean():
    caustics = torch.tensor([0.2, -0.2, 0.0]).view(3, 1, 1).expand(3, 4, 6).contiguous()
    panels = render_panels(
        torch.zeros(3, 4, 6),
        caustics,
        torch.zeros(3, 4, 6),
        torch.full((3, 4, 6), 0.5),
        torch.zeros(3, 4, 6),
        torch.ones(1, 4, 6),
        gain=2.0,
    )
    assert len(panels) == 6
    assert panels[3].shape == (4, 6, 3)
    # Channel mean is 0, so the panel sits at mid-gray and has no color.
    assert np.all(panels[3] == 128)
    assert panels[3][..., 0].tolist() == panels[3][..., 1].tolist() == panels[3][..., 2].tolist()


def test_six_panel_layout_matches_the_figure_order():
    palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    panels = [np.full((8, 16, 3), color, np.uint8) for color in palette]
    composer = PanelComposer((16, 8))
    image = composer(panels)
    bar = title_bar_height(8)
    assert image.shape == (composer.out_h, composer.out_w, 3)
    assert image[0, 0].tolist() == [0, 0, 0]
    assert image[:bar].max() == 255
    assert image[-bar:].max() == 255
    for row in range(2):
        for col in range(3):
            y = bar + row * 8 + 4
            x = col * 16 + 8
            assert image[y, x].tolist() == list(palette[row * 3 + col])


def test_resize_frame_uses_area_down_and_linear_up():
    source = np.arange(20 * 40 * 3, dtype=np.uint8).reshape(20, 40, 3)
    down = resize_frame(source, 10, 8)
    assert down.shape == (8, 10, 3)
    assert resize_frame(down, 10, 8) is down
    assert resize_frame(down, 20, 16).shape == (16, 20, 3)


def test_rectifier_skips_none_and_rejects_the_wrong_resolution():
    assert make_rectifier("none", (64, 32)) is None
    assert make_rectifier(None, (64, 32)) is None
    with pytest.raises(SystemExit, match="gopro_hero_10"):
        make_rectifier("gopro_hero_10", (512, 256))


def test_load_trained_model_does_not_reload_encoder_weights(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    from caustics_backscatter.config import load_config
    from caustics_backscatter.models import SMPDensePredictor
    from caustics_backscatter.registry import MODELS

    model = SMPDensePredictor(encoder_name="resnet18", encoder_weights=None, out_channels=3)
    cfg = load_config("configs/causticsnet.yaml", ["model.encoder_name=resnet18", "model.encoder_weights=null"])
    saved = OmegaConf.to_container(cfg, resolve=True)
    saved["model"]["encoder_weights"] = "imagenet"
    path = tmp_path / "step_000004.pt"
    torch.save({"step": 4, "model": model.state_dict(), "config": saved}, path)

    seen = {}
    real_build = MODELS.build

    def spy(section, **extra):
        container = OmegaConf.to_container(section, resolve=True) if not isinstance(section, dict) else section
        seen["encoder_weights"] = container["encoder_weights"]
        return real_build(section, **extra)

    monkeypatch.setattr(MODELS, "build", spy)
    loaded, loaded_cfg, step = load_trained_model(path, torch.device("cpu"))
    assert step == 4
    assert seen["encoder_weights"] is None
    assert loaded_cfg.data.image_size == cfg.data.image_size
    assert all(torch.equal(a, b) for a, b in zip(loaded.state_dict().values(), model.state_dict().values()))


def _write_color_video(path, color: tuple[int, int, int], frames: int, size: tuple[int, int], fps: float) -> None:
    import imageio_ffmpeg

    width, height = size
    writer = imageio_ffmpeg.write_frames(
        str(path), (width, height), fps=fps, quality=8, macro_block_size=1, ffmpeg_log_level="error"
    )
    writer.send(None)
    image = np.full((height, width, 3), color, dtype=np.uint8)
    for _ in range(frames):
        writer.send(image)
    writer.close()


def _read_video(path):
    from caustics_backscatter.infer import open_video

    meta, reader = open_video(path, None, 0.0, None, 1)
    width, height = (int(v) for v in meta["size"])
    frames = []
    try:
        for buffer in reader:
            frames.append(np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3).copy())
    finally:
        reader.close()
    return frames


def test_video_panels_remove_backscatter_from_the_caustics_cleaned_frame(tmp_path):
    color = (180, 40, 20)
    video = tmp_path / "clip.mp4"
    output = tmp_path / "panels.mp4"
    _write_color_video(video, color, frames=3, size=(64, 32), fps=10)
    caustics_net = _Caustics(0.25)
    backscatter_net = _Backscatter((0.2, 0.4, 0.6), beta=1.0)
    count = run_video(
        video,
        output,
        caustics_net,
        backscatter_net,
        _Depth(1.0),
        image_size=(32, 64),
        gain=2.0,
        camera_profile="none",
        batch_size=8,
        device=torch.device("cpu"),
    )
    assert count == 3
    # ffmpeg's yuv420 roundtrip shifts the flat color by a level or two; the networks must
    # still see that decoded RGB, not the caustics-cleaned frame.
    rgb = backscatter_net.seen[:, 0]
    assert torch.equal(caustics_net.seen[:, 0], rgb)
    assert torch.allclose(rgb, torch.tensor(color, dtype=torch.float32).view(1, 3, 1, 1) / 255.0, atol=4 / 255)
    cleaned = (rgb - 0.25).clamp(0, 1)
    assert not torch.allclose(rgb[:, :, :1, :1], cleaned[:, :, :1, :1])

    frames = _read_video(output)
    assert len(frames) == 3
    image = frames[0]
    bar = title_bar_height(32)
    assert image.shape == (32 * 2 + bar * 2, 64 * 3, 3)

    def pixel(col: int, row: int) -> np.ndarray:
        y = bar + row * 32 + 16
        x = col * 64 + 32
        return image[y, x]

    rgb_px = rgb[0, :, 0, 0]
    cleaned_px = cleaned[0, :, 0, 0]
    assert np.max(np.abs(pixel(0, 0).astype(int) - (rgb_px * 255).round().numpy())) <= 8
    factor = 1.0 - math.exp(-1.0)
    veil = torch.tensor([0.2, 0.4, 0.6]) * factor
    caustics_cleaned = (cleaned_px * 255).round()
    backscatter_cleaned = ((cleaned_px - veil + veil.mean()).clamp(0, 1) * 255).round()
    veil_u8 = (veil * 255).round()
    assert np.max(np.abs(pixel(1, 0).astype(int) - caustics_cleaned.numpy())) <= 8
    assert np.max(np.abs(pixel(2, 0).astype(int) - backscatter_cleaned.numpy())) <= 8
    assert np.max(np.abs(pixel(0, 1).astype(int) - 255)) <= 8
    assert np.max(np.abs(pixel(1, 1).astype(int) - veil_u8.numpy())) <= 8
