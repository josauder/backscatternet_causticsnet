import pytest

from caustics_backscatter.data.annotations import Window, clip_windows, parse_time, parse_windows, read_annotations

HEADER = "\ufefffilename,fw/bw,upside_down,transects,cut,comment,place\n"


def test_parse_time():
    assert parse_time("0:38") == 38
    assert parse_time("04:02") == 242
    assert parse_time("1:02:03") == 3723
    for bad in ("4", "1:75", "a:bc"):
        with pytest.raises(ValueError):
            parse_time(bad)


def test_parse_windows():
    assert parse_windows("begin-03:07, 04:48-end") == [Window(0.0, 187.0), Window(288.0, None)]
    with pytest.raises(ValueError):
        parse_windows("03:00-02:00")
    with pytest.raises(ValueError):
        parse_windows("03:00")


def test_clip_windows_merges_clips_and_drops_short():
    windows = [Window(50.0, None), Window(0.0, 20.0), Window(10.0, 30.0), Window(40.0, 45.0), Window(500.0, None)]
    assert clip_windows(windows, duration_s=100.0, min_segment_s=10.0) == [(0.0, 30.0), (50.0, 100.0)]


def test_read_annotations(tmp_path):
    rows = [
        "a/t.MP4,fw,,0:10-01:00,,forward transect,x",  # line 2: transect
        "a/r.MP4,,,,,free roaming,",  # line 3: roaming, place filled by the duplicate below
        ",,,,,,",  # line 4: separator
        "a/n.MP4,,,,,straight line,x",  # line 5: neither
        "a/x.MP4,,,,exclude,roaming,x",  # line 6: excluded
        "a/u.MP4,,1,0:10-01:00,,transect,x",  # line 7: upside down
        "a/s.MP4,,,,,short roam,x",  # line 8: not explicitly roaming
        "TODO sudan,,,,,roaming,x",  # line 9: not a video
        'a/t.MP4,fw,,"02:00-end",,,x',  # line 10: duplicate, merged
        "a/late.MP4,,,,,roaming,x",  # line 11: roaming after max_roaming_line
        "a/late_t.MP4,,,begin-0:30,,roaming,x",  # line 12: transects always count
        "a/r.MP4,,,,,roaming,Eilat",  # line 13: roaming after max_roaming_line, not merged
        "a/r.MP4,,,0:00-0:20,,,Eilat",  # line 14: duplicate transect row, merged
    ]
    path = tmp_path / "videos.csv"
    path.write_text(HEADER + "\n".join(rows), encoding="utf-8")
    videos = {v.filename: v for v in read_annotations(path, max_roaming_line=10)}
    assert sorted(videos) == ["a/late_t.MP4", "a/r.MP4", "a/t.MP4"]
    assert videos["a/t.MP4"].csv_line == 2
    assert videos["a/t.MP4"].selection == "transect"
    assert videos["a/t.MP4"].windows == [Window(10.0, 60.0), Window(120.0, None)]
    assert videos["a/t.MP4"].place == "x"
    assert videos["a/r.MP4"].selection == "roaming+transect"
    assert videos["a/r.MP4"].windows == [Window(0.0, None), Window(0.0, 20.0)]
    assert videos["a/r.MP4"].place == "Eilat"
    assert videos["a/late_t.MP4"].selection == "transect"
