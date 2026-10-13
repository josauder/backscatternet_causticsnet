"""Extract undistorted, resized video frames at a fixed frame rate.

For every segment (selected time window) of every video in the manifest written by
``select_videos``:

1. ffmpeg seeks to the segment start and resamples the segment to exactly ``--fps``
   frames per second by timestamp (deepreefmap's ``prepare_dataset`` uses an integer
   frame stride instead, which only gives 20 fps for 20/40/60 fps sources).
2. Each full-resolution frame is undistorted with deepreefmap's ``Rectifier`` for the
   manifest's camera profile, exactly as deepreefmap does for SC-SfMLearner training.
3. The frame is resized to ``--width`` x ``--height`` with ``INTER_AREA`` and written as
   JPEG at ``--jpeg-quality`` (default 100) with 4:4:4 sampling, so chroma is not
   subsampled; ``cam.txt`` holds the linear intrinsics scaled to that size.

Output layout (compatible with deepreefmap's ``ImageSequenceDataset``)::

    <out-dir>/manifest.json
    <out-dir>/sequences/<segment sequence>/000000.jpg ... cam.txt meta.json

Each segment is written to a hidden temporary directory that is renamed into place once
``meta.json`` is written, so a sequence directory only appears when it is complete. The job
can be re-run after interruptions, and training can read the finished sequences while
extraction is still running. Validation segments are extracted first.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


def _iter_resampled_frames(path: Path, fps: float, threads: int, start_s: float, duration_s: float):
    import imageio_ffmpeg

    # Seeking before -i is frame-accurate when decoding, and output timestamps restart at
    # 0, so the fps grid starts at the segment start.
    reader = imageio_ffmpeg.read_frames(
        str(path),
        input_params=["-threads", str(threads), "-ss", f"{start_s:.3f}"],
        output_params=["-t", f"{duration_s:.3f}", "-vf", f"fps={fps}"],
    )
    meta = next(reader)
    width, height = meta["size"]
    for buffer in reader:
        yield np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)


def iter_rectified_frames(
    path: Path,
    camera_profile: str,
    fps: float,
    threads: int,
    start_s: float,
    end_s: float,
    width: int,
    height: int,
):
    """Frames of ``[start_s, end_s)`` at ``fps``, undistorted at full resolution and resized
    to ``width`` x ``height`` with ``INTER_AREA`` (RGB uint8)."""
    from deepreefmap.camera.intrinsics import CameraProfile
    from deepreefmap.camera.rectification import Rectifier

    profile = CameraProfile.load(camera_profile)
    rectifier = Rectifier(profile)
    for frame in _iter_resampled_frames(path, fps, threads, start_s, end_s - start_s):
        if (frame.shape[1], frame.shape[0]) != tuple(profile.image_size):
            raise RuntimeError(
                f"{path}: frame size {frame.shape[1]}x{frame.shape[0]} does not match "
                f"camera profile {camera_profile} {profile.image_size}"
            )
        yield cv2.resize(rectifier.rectify(frame), (width, height), interpolation=cv2.INTER_AREA)


def extract_sequence(
    video: dict,
    segment: dict,
    sequences_dir: Path,
    camera_profile: str,
    width: int,
    height: int,
    fps: float,
    jpeg_quality: int,
    threads: int,
) -> tuple[str, int]:
    from deepreefmap.camera.intrinsics import CameraProfile, scale_intrinsics

    name = segment["sequence"]
    final_dir = sequences_dir / name
    if (final_dir / "meta.json").exists():
        return name, json.loads((final_dir / "meta.json").read_text())["num_frames"]

    tmp_dir = sequences_dir / f".{name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    profile = CameraProfile.load(camera_profile)
    start_s, end_s = float(segment["start_s"]), float(segment["end_s"])
    count = 0
    frames = iter_rectified_frames(Path(video["path"]), camera_profile, fps, threads, start_s, end_s, width, height)
    for resized in frames:
        ok = cv2.imwrite(
            str(tmp_dir / f"{count:06d}.jpg"),
            cv2.cvtColor(resized, cv2.COLOR_RGB2BGR),
            [
                cv2.IMWRITE_JPEG_QUALITY,
                jpeg_quality,
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
                cv2.IMWRITE_JPEG_OPTIMIZE,
                1,
            ],
        )
        if not ok:
            raise RuntimeError(f"Failed to write frame {count} of {video['path']}")
        count += 1
    if count == 0:
        shutil.rmtree(tmp_dir)
        raise RuntimeError(f"No frames decoded from {video['path']} [{start_s:.1f}s, {end_s:.1f}s]")

    intrinsics = scale_intrinsics(profile.k, profile.image_size, (width, height))
    np.savetxt(tmp_dir / "cam.txt", intrinsics, fmt="%.8f")
    meta = {
        "sequence": name,
        "video": video["sequence"],
        "source": video["path"],
        "source_start_s": start_s,
        "source_end_s": end_s,
        "csv_line": video.get("csv_line"),
        "selection": video.get("selection"),
        "source_fps": video.get("fps"),
        "fps": fps,
        "num_frames": count,
        "width": width,
        "height": height,
        "camera_profile": camera_profile,
        "jpeg_quality": jpeg_quality,
        "jpeg_sampling": "444",
        "intrinsics": intrinsics.tolist(),
    }
    (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)
    return name, count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=100,
        help="JPEG quality 0-100. Frames are stored 4:4:4 (no chroma subsampling).",
    )
    parser.add_argument("--workers", type=int, default=4, help="Segments processed in parallel")
    parser.add_argument("--ffmpeg-threads", type=int, default=4, help="Decoder threads per segment")
    parser.add_argument("--index", type=int, default=None, help="Only process this manifest entry (array jobs)")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    out_dir = args.out_dir.resolve()
    sequences_dir = out_dir / "sequences"
    sequences_dir.mkdir(parents=True, exist_ok=True)
    if args.manifest.resolve() != (out_dir / "manifest.json"):
        shutil.copyfile(args.manifest, out_dir / "manifest.json")

    videos = manifest["videos"]
    if args.index is not None:
        videos = [videos[args.index]]
    jobs = [(video, segment) for video in videos for segment in video["segments"]]
    jobs.sort(key=lambda job: job[0]["split"] != "val")

    kwargs = dict(
        sequences_dir=sequences_dir,
        camera_profile=manifest["camera_profile"],
        width=args.width,
        height=args.height,
        fps=args.fps,
        jpeg_quality=args.jpeg_quality,
        threads=args.ffmpeg_threads,
    )
    failures = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(extract_sequence, video, segment, **kwargs): segment for video, segment in jobs}
        for future in as_completed(futures):
            segment = futures[future]
            try:
                name, count = future.result()
                print(f"{name}: {count} frames", file=sys.stderr)
            except Exception as exc:
                failures += 1
                print(f"FAILED {segment['sequence']}: {exc}", file=sys.stderr)
    if failures:
        raise SystemExit(f"{failures} segment(s) failed")


if __name__ == "__main__":
    main()
