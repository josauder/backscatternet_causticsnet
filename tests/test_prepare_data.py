"""Selection from the annotation CSV and per-segment extraction on a synthetic 1920x1080 video."""

import json

import cv2
import numpy as np
import torch

from caustics_backscatter.data.annotations import read_annotations
from caustics_backscatter.data.dataset import labels_dir
from caustics_backscatter.data.prepare_frames import extract_sequence
from caustics_backscatter.data.segment_frames import Segmenter, pending_sequences, segment_sequences
from caustics_backscatter.data.select_videos import select

SOURCE_FPS = 30


def _write_video(path, seconds: int) -> None:
    import imageio_ffmpeg

    writer = imageio_ffmpeg.write_frames(
        str(path), (1920, 1080), fps=SOURCE_FPS, quality=8, macro_block_size=8, ffmpeg_log_level="error"
    )
    writer.send(None)
    for i in range(seconds * SOURCE_FPS):
        # Brightness encodes the source time so the extracted window can be checked.
        writer.send(np.full((1080, 1920, 3), 2 + 2 * i, dtype=np.uint8).tobytes())
    writer.close()


def test_select_and_extract_segment(tmp_path):
    root = tmp_path / "archive"
    (root / "site").mkdir(parents=True)
    _write_video(root / "site" / "v.mp4", seconds=4)
    csv_path = tmp_path / "videos.csv"
    csv_path.write_text("filename,upside_down,transects,cut,comment\nsite/v.mp4,,0:01-0:02,,transect\n")

    videos = select(
        root,
        read_annotations(csv_path),
        num_videos=1,
        num_val=0,
        seed=0,
        width=1920,
        height=1080,
        min_fps=20,
        min_duration_s=1,
        min_segment_s=1,
    )
    (video,) = videos
    assert video["selection"] == "transect" and video["split"] == "train"
    assert video["segments"] == [{"sequence": "site__v__seg0", "start_s": 1.0, "end_s": 2.0}]

    name, count = extract_sequence(
        video,
        video["segments"][0],
        tmp_path / "sequences",
        camera_profile="gopro_hero_10",
        width=128,
        height=64,
        fps=20.0,
        jpeg_quality=95,
        threads=1,
    )
    assert (name, count) == ("site__v__seg0", 20)
    directory = tmp_path / "sequences" / name
    meta = json.loads((directory / "meta.json").read_text())
    assert meta["source_start_s"] == 1.0 and meta["source_end_s"] == 2.0 and meta["num_frames"] == 20
    first = cv2.imread(str(directory / "000000.jpg"))[24:40, 56:72].mean()
    assert abs(first - (2 + 2 * SOURCE_FPS)) < 4


class _FakeSegmentationModel(torch.nn.Module):
    """Predicts fish (9) on the left half and sand (5) on the right half."""

    def __init__(self) -> None:
        super().__init__()
        self.input_shapes = []

    def forward(self, x):
        self.input_shapes.append(tuple(x.shape))
        logits = torch.zeros(x.shape[0], 40, *x.shape[-2:])
        half = x.shape[-1] // 2
        logits[:, 9, :, :half] = 1
        logits[:, 5, :, half:] = 1
        return logits


def test_segment_frames(tmp_path):
    root = tmp_path / "data"
    (root / "videos").mkdir(parents=True)
    _write_video(root / "videos" / "v.mp4", seconds=2)
    segment = {"sequence": "v__seg0", "start_s": 0.5, "end_s": 1.5}
    video = {"sequence": "v", "path": str(root / "videos" / "v.mp4"), "split": "val", "segments": [segment]}
    (root / "manifest.json").write_text(json.dumps({"videos": [video], "camera_profile": "gopro_hero_10"}))
    extract_sequence(video, segment, root / "sequences", "gopro_hero_10", 128, 64, 20.0, 95, 1)

    model = _FakeSegmentationModel()
    segmenter = Segmenter(model, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], amp=False)
    kwargs = dict(tag="t", input_size=(128, 256), batch_size=8, workers=0, ffmpeg_threads=1, model_info={"hf_repo": "x"})
    jobs, labeled, total = pending_sequences(root, "t")
    assert ([name for name, _ in jobs], labeled, total) == (["v__seg0"], 0, 1)
    assert segment_sequences(root, jobs, segmenter, **kwargs) == []
    assert set(model.input_shapes) == {(8, 3, 128, 256), (4, 3, 128, 256)}

    directory = root / "sequences" / "v__seg0"
    out = labels_dir(directory, "t")
    assert sorted(p.name for p in directory.iterdir() if p.name.startswith(".")) == []
    assert len(list(out.glob("*.png"))) == 20
    labels = cv2.imread(str(out / "000019.png"), cv2.IMREAD_UNCHANGED)
    assert labels.shape == (64, 128) and labels.dtype == np.uint8
    assert (labels[:, :64] == 9).all() and (labels[:, 64:] == 5).all()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["num_frames"] == 20 and meta["hf_repo"] == "x" and meta["id2label"]["9"] == "fish"
    assert pending_sequences(root, "t") == ([], 1, 1)

    # A sequence whose re-decoded frames do not match the extracted ones fails without output.
    frames_meta = json.loads((directory / "meta.json").read_text())
    (directory / "meta.json").write_text(json.dumps({**frames_meta, "num_frames": 21}))
    jobs, _, _ = pending_sequences(root, "u")
    assert segment_sequences(root, jobs, segmenter, **{**kwargs, "tag": "u"}) == ["v__seg0"]
    assert not labels_dir(directory, "u").exists() and not (directory / ".labels_u.tmp").exists()
