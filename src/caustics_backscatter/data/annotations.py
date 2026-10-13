"""Which videos (and which time windows of them) to use, from the coral archive's annotation CSV.

The CSV (``csv_videos_timestamp_export_*.csv``) has one row per video with ``filename``
relative to the archive root, ``transects`` (comma-separated ``start-end`` windows such as
``begin-03:07,04:48-11:21``), ``comment``, ``cut``, ``upside_down`` and ``place``. A video is
used if

* it has transect windows: only those windows are used, or
* its comment says "roaming" and its row is on line ``<= max_roaming_line`` of the raw
  file (the header is line 1): the whole video is used.

Rows with ``cut == "exclude"``, a non-empty ``upside_down``, an empty filename or a
non-video filename are skipped. Rows naming the same file are merged (union of windows).
"""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

VIDEO_SUFFIXES = (".mp4", ".mov")
_TIME = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2})$")


@dataclass(frozen=True)
class Window:
    start_s: float
    end_s: float | None  # None = until the end of the video


@dataclass
class AnnotatedVideo:
    filename: str
    csv_line: int
    selection: str  # "transect", "roaming" or "transect+roaming" for merged rows
    windows: list[Window] = field(default_factory=list)
    place: str = ""  # the CSV ``place`` column (first non-empty value of merged rows)


def parse_time(text: str) -> float:
    """``m:ss``, ``mm:ss`` or ``h:mm:ss`` to seconds."""
    match = _TIME.match(text.strip())
    if match is None:
        raise ValueError(f"Invalid time '{text}'")
    hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
    if seconds >= 60:
        raise ValueError(f"Invalid time '{text}'")
    return float(hours * 3600 + minutes * 60 + seconds)


def parse_windows(text: str) -> list[Window]:
    """``begin-03:07,04:48-11:21,12:00-end`` to windows; ``begin`` is 0, ``end`` the video end."""
    windows = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        start_text, sep, end_text = part.partition("-")
        if not sep:
            raise ValueError(f"Invalid window '{part}'")
        start_text, end_text = start_text.strip().lower(), end_text.strip().lower()
        start = 0.0 if start_text == "begin" else parse_time(start_text)
        end = None if end_text == "end" else parse_time(end_text)
        if end is not None and end <= start:
            raise ValueError(f"Window '{part}' ends before it starts")
        windows.append(Window(start, end))
    return windows


def is_roaming(comment: str) -> bool:
    return "roaming" in comment.lower()


def read_annotations(csv_path: str | Path, max_roaming_line: int = 1000) -> list[AnnotatedVideo]:
    """Videos selected by the transect / roaming rule, in CSV order."""
    videos: dict[str, AnnotatedVideo] = {}
    skipped = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            line = reader.line_num
            filename = (row.get("filename") or "").strip()
            if not filename.lower().endswith(VIDEO_SUFFIXES):
                continue
            if (row.get("cut") or "").strip().lower() == "exclude" or (row.get("upside_down") or "").strip():
                skipped += 1
                continue
            transects = (row.get("transects") or "").strip()
            if transects:
                try:
                    windows = parse_windows(transects)
                except ValueError as exc:
                    print(f"  skip (line {line}, {exc}): {filename}", file=sys.stderr)
                    skipped += 1
                    continue
                selection = "transect"
            elif is_roaming(row.get("comment") or "") and line <= max_roaming_line:
                windows = [Window(0.0, None)]
                selection = "roaming"
            else:
                continue
            place = (row.get("place") or "").strip()
            if filename in videos:
                video = videos[filename]
                video.windows.extend(windows)
                if selection not in video.selection.split("+"):
                    video.selection = f"{video.selection}+{selection}"
                if not video.place:
                    video.place = place
            else:
                videos[filename] = AnnotatedVideo(filename, line, selection, windows, place)
    print(f"{len(videos)} videos selected from {csv_path} ({skipped} rows excluded)", file=sys.stderr)
    return list(videos.values())


def clip_windows(windows: list[Window], duration_s: float, min_segment_s: float) -> list[tuple[float, float]]:
    """Windows clipped to ``[0, duration_s]``, overlaps merged, short ones dropped, sorted."""
    spans = sorted(
        (w.start_s, duration_s if w.end_s is None else min(w.end_s, duration_s))
        for w in windows
        if w.start_s < duration_s
    )
    merged: list[list[float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged if end - start >= min_segment_s]
