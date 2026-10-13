"""Run CausticsNet and BackscatterNet on a video and write a six-panel mp4.

The panels match the paper figure::

    Input Video          Caustics Cleaned       Backscatter Cleaned
    Estimated Caustics   Estimated Backscatter  Estimated Depth

CausticsNet, BackscatterNet and the frozen SC-SfMLearner all see the original
RGB frame. The veil ``B = B_inf (1 - exp(-beta Z))`` is then subtracted from
the caustics-cleaned frame, so the third panel has both effects removed. For
display, the per-pixel channel mean of ``B`` is added back (as in the training
visualization) so the color cast is removed and the brightness stays.

Frames are undistorted with a deepreefmap camera profile and resized to the
training resolution the same way as ``prepare_frames``.

    uv run python scripts/infer_video.py VIDEO -o panels.mp4 \\
        --caustics runs/causticsnet/checkpoints/latest.pt \\
        --backscatter runs/backscatternet/checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import functools
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import caustics_backscatter.geometry  # noqa: F401  (registers the geometry provider)
import caustics_backscatter.models  # noqa: F401  (registers both networks)
from caustics_backscatter.models.backscatternet import backscatter
from caustics_backscatter.registry import GEOMETRY_PROVIDERS, MODELS
from caustics_backscatter.utils.viz import depth_to_uint8, signed_to_uint8, to_uint8

TOP_TITLES = ("Input Video", "Caustics Cleaned", "Backscatter Cleaned")
BOTTOM_TITLES = ("Estimated Caustics", "Estimated Backscatter", "Estimated Depth")

# Bold sans-serif, closest common face to the paper figure. The first file that exists is used.
_BOLD_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)


def default_checkpoint(run_name: str) -> Path:
    root = Path(os.environ.get("CB_RUNS_ROOT", "runs"))
    return root / run_name / "checkpoints" / "latest.pt"


def load_trained_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, DictConfig, int]:
    """Load a ``latest.pt`` / ``step_*.pt`` checkpoint. Encoder weights are not re-downloaded."""
    if not path.is_file():
        raise SystemExit(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in state or "config" not in state:
        raise SystemExit(f"{path} is not a training checkpoint (expected keys 'model' and 'config')")
    cfg = OmegaConf.create(state["config"])
    # The checkpoint already contains the encoder; skip the ImageNet download.
    cfg.model.encoder_weights = None
    model = MODELS.build(cfg.model)
    model.load_state_dict(state["model"])
    model.eval().requires_grad_(False)
    model.to(device)
    return model, cfg, int(state.get("step", 0))


def load_geometry(cfg: DictConfig, device: torch.device, checkpoint: Path | None) -> torch.nn.Module:
    geo = OmegaConf.create(OmegaConf.to_container(cfg.geometry, resolve=True))
    if checkpoint is not None:
        geo.checkpoint = str(checkpoint)
    provider = GEOMETRY_PROVIDERS.build(geo).to(device)
    provider.eval()
    return provider


def make_rectifier(camera_profile: str | None, frame_size: tuple[int, int]):
    """Undistort ``frame_size`` ``(W, H)`` frames, or return None when rectification is off.

    A profile whose calibrated resolution does not match the video is an error: the
    training frames were undistorted with that profile before the networks ever saw them.
    """
    if camera_profile is None or camera_profile.lower() == "none":
        return None
    from deepreefmap.camera.intrinsics import CameraProfile
    from deepreefmap.camera.rectification import Rectifier

    profile = CameraProfile.load(camera_profile)
    expected = tuple(int(v) for v in profile.image_size)
    if tuple(frame_size) != expected:
        width, height = frame_size
        expected_w, expected_h = expected
        raise SystemExit(
            f"Video is {width}x{height}, but camera profile {camera_profile!r} expects "
            f"{expected_w}x{expected_h}. Pass --camera-profile none to skip undistortion."
        )
    return Rectifier(profile)


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Match ``prepare_frames``: ``INTER_AREA`` when shrinking, ``INTER_LINEAR`` when growing."""
    src_h, src_w = frame.shape[:2]
    if (src_w, src_h) == (width, height):
        return frame
    interpolation = cv2.INTER_AREA if src_w > width or src_h > height else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, height), interpolation=interpolation)


@torch.inference_mode()
def infer_batch(
    rgb_u8: torch.Tensor,
    caustics_net: torch.nn.Module,
    backscatter_net: torch.nn.Module,
    geometry: torch.nn.Module,
    *,
    keep_brightness: bool = True,
) -> dict[str, torch.Tensor]:
    """Run both networks on one batch of RGB frames.

    ``rgb_u8`` is ``[B, 3, H, W]`` uint8. Caustics, backscatter parameters and depth are
    all predicted from this RGB batch. The veil is subtracted from the caustics-cleaned
    frame. With ``keep_brightness``, the per-pixel channel mean of the veil is added
    back so the panel keeps its brightness.
    """
    rgb = rgb_u8.float() / 255.0
    views = rgb[:, None]
    amp = rgb_u8.is_cuda
    with torch.autocast(device_type=rgb_u8.device.type, dtype=torch.bfloat16, enabled=amp):
        caustics = caustics_net(views).float()[:, 0]
        b_inf, beta = backscatter_net(views)
    b_inf = b_inf.float()[:, 0]
    beta = beta.float()[:, 0]
    depth = geometry.depth(rgb_u8)
    scatter = backscatter(b_inf, beta, depth)

    caustics_cleaned = (rgb - caustics).clamp(0, 1)
    removed = caustics_cleaned - scatter
    if keep_brightness:
        removed = removed + scatter.mean(dim=1, keepdim=True)
    return {
        "caustics": caustics,
        "caustics_cleaned": caustics_cleaned,
        "backscatter": scatter,
        "backscatter_cleaned": removed.clamp(0, 1),
        "depth": depth,
        "b_inf": b_inf,
        "beta": beta,
    }


def render_panels(
    rgb: torch.Tensor,
    caustics: torch.Tensor,
    caustics_cleaned: torch.Tensor,
    scatter: torch.Tensor,
    backscatter_cleaned: torch.Tensor,
    depth: torch.Tensor,
    gain: float,
) -> list[np.ndarray]:
    """Six ``[H, W, 3]`` uint8 panels, in figure order.

    Estimated caustics are drawn as the channel-mean residual around mid-gray (the paper
    figure is gray). Removal itself uses the full RGB caustic prediction.
    """
    gray = caustics.mean(dim=0, keepdim=True).expand(3, -1, -1)
    return [
        to_uint8(rgb),
        to_uint8(caustics_cleaned),
        to_uint8(backscatter_cleaned),
        signed_to_uint8(gray, gain),
        to_uint8(scatter),
        depth_to_uint8(depth),
    ]


def title_bar_height(panel_h: int) -> int:
    """Bar height matched to the paper figure (88 px over a 197 px panel)."""
    return max(28, round(panel_h * 88 / 197))


@functools.lru_cache(maxsize=1)
def _bold_font_path() -> str | None:
    return next((path for path in _BOLD_FONTS if Path(path).is_file()), None)


def _title_bar(width: int, height: int, titles: tuple[str, str, str]) -> np.ndarray:
    font_path = _bold_font_path()
    if font_path is None:
        return _title_bar_cv2(width, height, titles)
    from PIL import Image, ImageDraw, ImageFont

    panel_w = width // 3
    # The longest title in the figure fills about 88% of its column.
    target = panel_w * 0.88
    size = max(12, round(height * 0.34))
    font = ImageFont.truetype(font_path, size=size)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for _ in range(size):
        widest = max(probe.textlength(title, font=font) for title in titles)
        if widest <= target or size <= 12:
            break
        size -= 1
        font = ImageFont.truetype(font_path, size=size)

    bar = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(bar)
    for index, title in enumerate(titles):
        bbox = draw.textbbox((0, 0), title, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = index * panel_w + (panel_w - text_w) / 2 - bbox[0]
        y = (height - text_h) / 2 - bbox[1]
        draw.text((x, y), title, font=font, fill=(255, 255, 255))
    return np.asarray(bar, dtype=np.uint8)


def _title_bar_cv2(width: int, height: int, titles: tuple[str, str, str]) -> np.ndarray:
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    panel_w = width // 3
    font = cv2.FONT_HERSHEY_DUPLEX
    target = panel_w * 0.88
    scale = max(0.4, height / 70)
    thickness = 1
    for _ in range(12):
        widest = max(cv2.getTextSize(title, font, scale, thickness)[0][0] for title in titles)
        if widest <= target or scale <= 0.4:
            break
        scale *= target / widest
    thickness = max(1, round(scale * 1.4))
    for index, title in enumerate(titles):
        (text_w, text_h), _ = cv2.getTextSize(title, font, scale, thickness)
        x = index * panel_w + (panel_w - text_w) // 2
        y = (height + text_h) // 2
        cv2.putText(bar, title, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return bar


class PanelComposer:
    """Stack the six panels into the labeled figure. Titles are rendered once."""

    def __init__(self, panel_size: tuple[int, int], scale: float = 1.0) -> None:
        """``panel_size`` is ``(W, H)`` of one network-resolution panel."""
        src_w, src_h = panel_size
        self.src_w = src_w
        self.src_h = src_h
        self.panel_w = max(1, round(src_w * scale))
        self.panel_h = max(1, round(src_h * scale))
        self.bar_h = title_bar_height(self.panel_h)
        width = self.panel_w * 3
        self.top = _title_bar(width, self.bar_h, TOP_TITLES)
        self.bottom = _title_bar(width, self.bar_h, BOTTOM_TITLES)
        self.out_w = width + (width % 2)
        self.out_h = self.panel_h * 2 + self.bar_h * 2
        self.out_h += self.out_h % 2

    def __call__(self, panels: list[np.ndarray]) -> np.ndarray:
        if len(panels) != 6:
            raise ValueError(f"Expected 6 panels, got {len(panels)}")
        sized = []
        for panel in panels:
            if panel.shape[:2] != (self.src_h, self.src_w):
                raise ValueError(f"Panel is {panel.shape[1]}x{panel.shape[0]}, expected {self.src_w}x{self.src_h}")
            if (self.panel_w, self.panel_h) != (self.src_w, self.src_h):
                interpolation = cv2.INTER_AREA if self.panel_w < self.src_w else cv2.INTER_LINEAR
                panel = cv2.resize(panel, (self.panel_w, self.panel_h), interpolation=interpolation)
            sized.append(panel)
        image = np.concatenate(
            [self.top, np.concatenate(sized[:3], axis=1), np.concatenate(sized[3:], axis=1), self.bottom],
            axis=0,
        )
        if image.shape[1] != self.out_w or image.shape[0] != self.out_h:
            image = np.pad(image, ((0, self.out_h - image.shape[0]), (0, self.out_w - image.shape[1]), (0, 0)))
        return np.ascontiguousarray(image)


def open_video(path: Path, fps: float | None, start_s: float, end_s: float | None, threads: int):
    """Start an ffmpeg reader. Returns ``(meta, reader)``; the caller closes ``reader``.

    ``meta["size"]`` is ``(W, H)`` of the frames the reader then yields as raw RGB bytes.
    """
    import imageio_ffmpeg

    input_params = ["-threads", str(threads)]
    output_params: list[str] = []
    if start_s > 0:
        input_params += ["-ss", f"{start_s:.3f}"]
    if end_s is not None:
        output_params += ["-t", f"{end_s - start_s:.3f}"]
    if fps is not None:
        output_params += ["-vf", f"fps={fps}"]
    reader = imageio_ffmpeg.read_frames(
        str(path), input_params=input_params, output_params=output_params or None
    )
    return next(reader), reader


def _output_fps(meta: dict, fps: float | None, start_s: float, end_s: float | None) -> tuple[float, int | None]:
    source_fps = float(meta.get("fps") or 0.0)
    out_fps = float(fps if fps is not None else source_fps)
    if out_fps <= 0:
        out_fps = 20.0
    duration = float(meta.get("duration") or 0.0)
    if end_s is not None:
        duration = end_s - start_s
    elif start_s > 0:
        duration = max(0.0, duration - start_s)
    total = int(round(duration * out_fps)) if duration > 0 else None
    return out_fps, total


def run_video(
    video: Path,
    output: Path,
    caustics_net: torch.nn.Module,
    backscatter_net: torch.nn.Module,
    geometry: torch.nn.Module,
    *,
    image_size: tuple[int, int],
    gain: float,
    camera_profile: str | None,
    fps: float | None = None,
    start_s: float = 0.0,
    end_s: float | None = None,
    batch_size: int = 8,
    max_frames: int | None = None,
    scale: float = 1.0,
    keep_brightness: bool = True,
    threads: int = 2,
    device: torch.device | None = None,
) -> int:
    """Write the six-panel video. ``image_size`` is ``(H, W)``. Returns the frame count."""
    if not video.is_file():
        raise SystemExit(f"Video not found: {video}")
    if end_s is not None and end_s <= start_s:
        raise SystemExit("--end must be greater than --start")
    device = device or torch.device("cpu")
    height, width = image_size
    meta, reader = open_video(video, fps, start_s, end_s, threads)
    src_w, src_h = (int(v) for v in meta["size"])
    try:
        rectifier = make_rectifier(camera_profile, (src_w, src_h))
        out_fps, total = _output_fps(meta, fps, start_s, end_s)
        if max_frames is not None:
            total = max_frames if total is None else min(total, max_frames)
        composer = PanelComposer((width, height), scale=scale)
        output.parent.mkdir(parents=True, exist_ok=True)
        import imageio_ffmpeg

        writer = imageio_ffmpeg.write_frames(
            str(output),
            (composer.out_w, composer.out_h),
            fps=out_fps,
            quality=8,
            macro_block_size=1,
            ffmpeg_log_level="error",
        )
        profile = "none" if rectifier is None else camera_profile
        print(
            f"{video.name}: {src_w}x{src_h} -> model {width}x{height}, rectify {profile}, "
            f"{out_fps:.3f} fps, device {device}, writing {output} ({composer.out_w}x{composer.out_h})",
            flush=True,
        )
        pending: list[np.ndarray] = []
        count = 0
        reported = False

        def flush(batch_frames: list[np.ndarray]) -> int:
            nonlocal reported
            rgb_u8 = torch.from_numpy(np.stack(batch_frames)).permute(0, 3, 1, 2).contiguous()
            outputs = infer_batch(
                rgb_u8.to(device, non_blocking=True),
                caustics_net,
                backscatter_net,
                geometry,
                keep_brightness=keep_brightness,
            )
            if not reported:
                b_inf = outputs["b_inf"][0].tolist()
                beta = outputs["beta"][0].tolist()
                print(
                    "b_inf [" + ", ".join(f"{v:.3f}" for v in b_inf) + "]  "
                    "beta [" + ", ".join(f"{v:.3f}" for v in beta) + "]",
                    flush=True,
                )
                reported = True
            for index in range(rgb_u8.shape[0]):
                panels = render_panels(
                    rgb_u8[index].float() / 255.0,
                    outputs["caustics"][index],
                    outputs["caustics_cleaned"][index],
                    outputs["backscatter"][index],
                    outputs["backscatter_cleaned"][index],
                    outputs["depth"][index],
                    gain,
                )
                writer.send(composer(panels))
            return len(batch_frames)

        progress = None
        try:
            writer.send(None)
            progress = tqdm(total=total, unit="frame", desc="infer")
            for buffer in reader:
                frame = np.frombuffer(buffer, dtype=np.uint8).reshape(src_h, src_w, 3).copy()
                if rectifier is not None:
                    frame = rectifier.rectify(frame)
                pending.append(resize_frame(frame, width, height))
                reached = max_frames is not None and count + len(pending) >= max_frames
                if len(pending) == batch_size or reached:
                    if max_frames is not None:
                        pending = pending[: max_frames - count]
                    written = flush(pending)
                    count += written
                    progress.update(written)
                    pending = []
                    if max_frames is not None and count >= max_frames:
                        break
            if pending:
                written = flush(pending)
                count += written
                progress.update(written)
        finally:
            if progress is not None:
                progress.close()
            writer.close()
    finally:
        reader.close()

    if count == 0:
        output.unlink(missing_ok=True)
        raise SystemExit(f"No frames decoded from {video}")
    print(f"Wrote {count} frames to {output}", flush=True)
    return count


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="Input video")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output mp4 (default: <stem>_panels.mp4)")
    parser.add_argument("--caustics", type=Path, default=None, help="CausticsNet checkpoint")
    parser.add_argument("--backscatter", type=Path, default=None, help="BackscatterNet checkpoint")
    parser.add_argument("--geometry-checkpoint", type=Path, default=None, help="SC-SfMLearner weights (default: download)")
    parser.add_argument(
        "--camera-profile",
        default="gopro_hero_10",
        help="Undistort with this deepreefmap profile before resizing. 'none' skips it. Default: gopro_hero_10.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Resample the video to this frame rate (default: source)")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--scale", type=float, default=1.0, help="Scale each panel before drawing the titles")
    parser.add_argument("--gain", type=float, default=None, help="Caustics display gain (default: viz.residual_gain)")
    parser.add_argument("--width", type=int, default=None, help="Model width (default: the checkpoint's training width)")
    parser.add_argument("--height", type=int, default=None, help="Model height (default: the checkpoint's training height)")
    parser.add_argument("--threads", type=int, default=2, help="ffmpeg decoder threads")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available)")
    parser.add_argument(
        "--keep-brightness",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add the veil's channel mean back onto the backscatter-cleaned panel (default: on).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    caustics_path = args.caustics or default_checkpoint("causticsnet")
    backscatter_path = args.backscatter or default_checkpoint("backscatternet")
    caustics_net, caustics_cfg, caustics_step = load_trained_model(caustics_path, device)
    backscatter_net, backscatter_cfg, backscatter_step = load_trained_model(backscatter_path, device)
    trained_h, trained_w = (int(v) for v in caustics_cfg.data.image_size)
    other_h, other_w = (int(v) for v in backscatter_cfg.data.image_size)
    if (other_h, other_w) != (trained_h, trained_w):
        raise SystemExit(
            f"CausticsNet was trained at {trained_w}x{trained_h} and BackscatterNet at {other_w}x{other_h}."
        )
    height = args.height or trained_h
    width = args.width or trained_w
    gain = float(caustics_cfg.viz.residual_gain if args.gain is None else args.gain)
    print(
        f"CausticsNet step {caustics_step} ({caustics_path}), "
        f"BackscatterNet step {backscatter_step} ({backscatter_path})",
        flush=True,
    )
    geometry = load_geometry(caustics_cfg, device, args.geometry_checkpoint)
    output = args.output or Path(f"{args.video.stem}_panels.mp4")
    run_video(
        args.video,
        output,
        caustics_net,
        backscatter_net,
        geometry,
        image_size=(height, width),
        gain=gain,
        camera_profile=args.camera_profile,
        fps=args.fps,
        start_s=args.start,
        end_s=args.end,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        scale=args.scale,
        keep_brightness=args.keep_brightness,
        threads=args.threads,
        device=device,
    )


if __name__ == "__main__":
    main()
