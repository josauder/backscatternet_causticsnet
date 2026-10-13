import copy

import pytest

from caustics_backscatter.data import select_videos
from caustics_backscatter.data.annotations import read_annotations

HEADER = "\ufefffilename,fw/bw,upside_down,transects,cut,comment,place\n"
FILTERS = dict(width=1920, height=1080, min_fps=20.0, min_duration_s=60.0, min_segment_s=10.0)

ROWS = [
    ("old/a.MP4", "Sudan"),
    ("old/b.MP4", "Eilat, Israel"),
    ("eilat_n/1.MP4", "Eilat, Israel"),
    ("eilat_n/2.MP4", "Eilat, Israel"),
    ("eilat_s/1.MP4", "Eilat, Israel"),
    ("aqaba/1.MP4", "Aqaba, Jordan"),
    ("aqaba/2.MP4", "Aqaba, Jordan"),
    ("aqaba/small.MP4", "Aqaba, Jordan"),
    ("sudan/1.MP4", "Sudan"),
    ("saudi/1.MP4", "Jeddah, Saudi Arabia"),
]


@pytest.fixture
def archive(tmp_path, monkeypatch):
    root = tmp_path / "coral"
    for filename, _ in ROWS:
        (root / filename).parent.mkdir(parents=True, exist_ok=True)
        (root / filename).touch()
    csv = tmp_path / "videos.csv"
    csv.write_text(HEADER + "\n".join(f'{name},fw,,begin-02:00,,transect,"{place}"' for name, place in ROWS))

    def probe(path):
        width = 1280 if path.name == "small.MP4" else 1920
        return {"fps": 30.0, "num_source_frames": 9000, "width": width, "height": 1080 * width // 1920, "duration_s": 300.0}

    monkeypatch.setattr(select_videos, "probe", probe)
    return root, read_annotations(csv)


def test_select_round_robin_over_sites(archive):
    root, annotations = archive
    videos = select_videos.select(root, annotations, num_videos=5, num_val=1, seed=0, **FILTERS)
    assert len({v["site"] for v in videos}) == 5
    assert [v["split"] for v in videos] == ["train"] * 4 + ["val"]
    assert all(v["place"] for v in videos)
    assert videos[0]["segments"] == [{"sequence": f"{videos[0]['sequence']}__seg0", "start_s": 0.0, "end_s": 120.0}]


def test_extend_appends_matching_new_videos(archive):
    root, annotations = archive
    old = [v for v in annotations if v.filename.startswith("old/")]
    old = select_videos.select(root, old, num_videos=2, num_val=1, seed=0, **FILTERS)
    for entry in old:
        del entry["place"]  # manifests written before the place column was read
    manifest = {"root": str(root), "camera_profile": "gopro_hero_10", "videos": old}
    before = copy.deepcopy(old)

    added = select_videos.extend(
        manifest, annotations, "eilat|jordan", num_extra=3, num_extra_val=0, seed=0, **FILTERS
    )

    places = {str(root / name): place for name, place in ROWS}
    assert len(added) == 3
    assert manifest["videos"][: len(before)] == [{**entry, "place": places[entry["path"]]} for entry in before]
    assert manifest["videos"][len(before) :] == added
    assert all(entry["split"] == "train" for entry in added)
    assert all(("Eilat" in entry["place"]) or ("Jordan" in entry["place"]) for entry in added)
    assert not {entry["path"] for entry in added} & {entry["path"] for entry in before}
    sites = [entry["site"] for entry in added]
    assert set(sites) <= {"eilat_n", "eilat_s", "aqaba"} and len(set(sites)) >= 2


def test_extend_takes_what_is_left(archive, capsys):
    root, annotations = archive
    manifest = {"root": str(root), "camera_profile": "gopro_hero_10", "videos": []}
    first = select_videos.extend(manifest, annotations, "jordan", num_extra=1, num_extra_val=0, seed=0, **FILTERS)
    rest = select_videos.extend(manifest, annotations, "jordan", num_extra=5, num_extra_val=1, seed=0, **FILTERS)
    assert len(first) == 1 and len(rest) == 1  # aqaba/small.MP4 has the wrong resolution
    assert rest[0]["split"] == "val"
    assert {e["path"] for e in manifest["videos"]} == {str(root / "aqaba/1.MP4"), str(root / "aqaba/2.MP4")}
    assert "WARNING: only 1 new videos" in capsys.readouterr().err
