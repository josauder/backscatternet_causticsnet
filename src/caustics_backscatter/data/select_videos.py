"""Pick a small, site-diverse set of annotated videos from the coral archive and write a manifest.

Candidates come from the annotation CSV (see ``annotations.py``): transect windows of
videos with transects, and whole videos whose comment says "roaming" on CSV line
``<= --max-roaming-line``. Videos are grouped by their top-level folder under ``--root``
(the dive site), sites and videos are shuffled with ``--seed``, and videos are taken
round-robin across sites so the selection spans as many sites as possible. Only videos
whose resolution matches the camera profile (1920x1080 for ``gopro_hero_10``), whose frame
rate allows 20 fps sampling and whose selected windows total at least ``--min-duration-s``
are accepted. Each window becomes one segment (sequence) of the manifest entry. The last
``--num-val`` selected videos form the validation split.

``--place-regex`` restricts the candidates to rows whose CSV ``place`` matches
(case-insensitive). With ``--extend``, the existing manifest at ``--out`` is kept as is and
up to ``--num-extra`` videos that are not in it yet are appended (as train, except the last
``--num-extra-val``), so sequence indices and finished sequences stay valid.
``--list-places`` prints the number of candidate videos per place and exits.

    python -m caustics_backscatter.data.select_videos \\
        --root ./videos --csv ./annotations.csv --out data/manifest.json
    python -m caustics_backscatter.data.select_videos \\
        --root ./videos --csv ./annotations.csv --out data/manifest.json \\
        --extend --place-regex 'reef' --num-extra 20 --seed 1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

import cv2

from caustics_backscatter.data.annotations import AnnotatedVideo, clip_windows, read_annotations


def probe(path: Path) -> dict | None:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if fps <= 0 or frames <= 0:
        return None
    return {"fps": fps, "num_source_frames": frames, "width": width, "height": height, "duration_s": frames / fps}


def sequence_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return re.sub(r"[^A-Za-z0-9_-]+", "_", "__".join(rel.parts))


def filter_places(annotations: list[AnnotatedVideo], place_regex: str | None) -> list[AnnotatedVideo]:
    if place_regex is None:
        return annotations
    pattern = re.compile(place_regex, re.IGNORECASE)
    return [video for video in annotations if pattern.search(video.place)]


def iter_accepted(
    root: Path,
    annotations: list[AnnotatedVideo],
    seed: int,
    width: int,
    height: int,
    min_fps: float,
    min_duration_s: float,
    min_segment_s: float,
) -> Iterator[dict]:
    """Manifest entries (without ``split``) of the videos passing the filters, round-robin over sites."""
    by_site: dict[str, list[AnnotatedVideo]] = defaultdict(list)
    for video in annotations:
        parts = Path(video.filename).parts
        by_site[parts[0] if len(parts) > 1 else "."].append(video)

    rng = random.Random(seed)
    sites = sorted(by_site)
    rng.shuffle(sites)
    for site in sites:
        rng.shuffle(by_site[site])
    print(f"{len(annotations)} annotated videos across {len(sites)} sites", file=sys.stderr)

    cursors = {site: 0 for site in sites}
    while any(cursors[s] < len(by_site[s]) for s in sites):
        for site in sites:
            if cursors[site] >= len(by_site[site]):
                continue
            video = by_site[site][cursors[site]]
            cursors[site] += 1
            path = root / video.filename
            if not path.is_file():
                print(f"  skip (missing): {path}", file=sys.stderr)
                continue
            info = probe(path)
            if info is None:
                print(f"  skip (unreadable): {path}", file=sys.stderr)
                continue
            if (info["width"], info["height"]) != (width, height):
                print(f"  skip ({info['width']}x{info['height']}): {path}", file=sys.stderr)
                continue
            if info["fps"] < min_fps:
                print(f"  skip (fps={info['fps']:.2f}): {path}", file=sys.stderr)
                continue
            spans = clip_windows(video.windows, info["duration_s"], min_segment_s)
            total_s = sum(end - start for start, end in spans)
            if total_s < min_duration_s:
                print(f"  skip ({total_s:.0f}s of {video.selection} footage): {path}", file=sys.stderr)
                continue
            sequence = sequence_name(root, path)
            yield {
                "path": str(path),
                "site": site,
                "place": video.place,
                "sequence": sequence,
                "csv_line": video.csv_line,
                "selection": video.selection,
                **info,
                "segments": [
                    {"sequence": f"{sequence}__seg{k}", "start_s": start, "end_s": end}
                    for k, (start, end) in enumerate(spans)
                ],
            }


def take(entries: Iterator[dict], count: int) -> list[dict]:
    """The first ``count`` entries; the generator is not advanced (no probing) past them."""
    selected: list[dict] = []
    if count <= 0:
        return selected
    for entry in entries:
        selected.append(entry)
        total_s = sum(s["end_s"] - s["start_s"] for s in entry["segments"])
        print(
            f"  selected [{len(selected)}/{count}] {entry['path']} ({entry['selection']}, "
            f"{len(entry['segments'])} segment(s), {total_s:.0f}s)",
            file=sys.stderr,
        )
        if len(selected) >= count:
            break
    return selected


def assign_splits(entries: list[dict], num_val: int, train: str = "train", val: str = "val") -> None:
    for i, entry in enumerate(entries):
        entry["split"] = val if i >= len(entries) - num_val else train


def select(
    root: Path,
    annotations: list[AnnotatedVideo],
    num_videos: int,
    num_val: int,
    seed: int,
    width: int,
    height: int,
    min_fps: float,
    min_duration_s: float,
    min_segment_s: float,
) -> list[dict]:
    if not annotations:
        raise RuntimeError("No annotated videos to select from")
    entries = iter_accepted(root, annotations, seed, width, height, min_fps, min_duration_s, min_segment_s)
    selected = take(entries, num_videos)
    if len(selected) < num_videos:
        raise RuntimeError(f"Only {len(selected)} videos matched the filters, wanted {num_videos}")
    assign_splits(selected, num_val)
    return selected


def extend(
    manifest: dict,
    annotations: list[AnnotatedVideo],
    place_regex: str | None,
    num_extra: int,
    num_extra_val: int,
    seed: int,
    width: int,
    height: int,
    min_fps: float,
    min_duration_s: float,
    min_segment_s: float,
) -> list[dict]:
    """Append up to ``num_extra`` accepted videos matching ``place_regex`` that are not in
    ``manifest`` yet (in place).

    Existing entries keep their position and split; they only get ``place`` backfilled from
    ``annotations``. Returns the appended entries.
    """
    root = Path(manifest["root"])
    places = {str(root / video.filename): video.place for video in annotations}
    for entry in manifest["videos"]:
        entry.setdefault("place", places.get(entry["path"], ""))

    paths = {entry["path"] for entry in manifest["videos"]}
    sequences = {entry["sequence"] for entry in manifest["videos"]}
    matching = filter_places(annotations, place_regex)
    candidates = [
        video
        for video in matching
        if str(root / video.filename) not in paths and sequence_name(root, root / video.filename) not in sequences
    ]
    print(
        f"{len(matching)} annotated videos match the place filter, {len(matching) - len(candidates)} "
        "of them are already in the manifest",
        file=sys.stderr,
    )
    entries = iter_accepted(root, candidates, seed, width, height, min_fps, min_duration_s, min_segment_s)
    added = take(entries, num_extra)
    if len(added) < num_extra:
        print(f"WARNING: only {len(added)} new videos matched the filters, wanted {num_extra}", file=sys.stderr)
    assign_splits(added, min(num_extra_val, len(added)))
    manifest["videos"].extend(added)
    return added


def list_places(annotations: list[AnnotatedVideo], root: Path, manifest: dict | None, place_regex: str | None) -> None:
    counts = Counter(video.place for video in annotations)
    in_manifest = Counter()
    if manifest is not None:
        paths = {entry["path"] for entry in manifest["videos"]}
        in_manifest = Counter(video.place for video in annotations if str(root / video.filename) in paths)
    matching = {video.place for video in filter_places(annotations, place_regex)} if place_regex else set()
    print(f"{'candidates':>10} {'in manifest':>11}  place")
    for place, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        mark = "  *" if place in matching else ""
        print(f"{count:>10} {in_manifest[place]:>11}  {place or '(empty)'}{mark}")
    if place_regex:
        print(f"* matches --place-regex {place_regex!r}")


def write_manifest(manifest: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Directory of source videos")
    parser.add_argument("--csv", type=Path, required=True, help="Annotation CSV (filenames relative to --root)")
    parser.add_argument("--out", type=Path, required=True, help="Path of the manifest JSON to write")
    parser.add_argument("--num-videos", type=int, default=10)
    parser.add_argument("--num-val", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-profile", default="gopro_hero_10")
    parser.add_argument("--min-fps", type=float, default=20.0)
    parser.add_argument("--min-duration-s", type=float, default=60.0, help="Minimum total selected footage per video")
    parser.add_argument("--min-segment-s", type=float, default=10.0, help="Shorter windows are dropped")
    parser.add_argument("--max-roaming-line", type=int, default=1000, help="Roaming videos only up to this CSV line")
    parser.add_argument("--place-regex", default=None, help="Only videos whose CSV place matches (case-insensitive)")
    parser.add_argument("--extend", action="store_true", help="Append videos to the existing manifest at --out")
    parser.add_argument("--num-extra", type=int, default=0, help="Videos to append with --extend")
    parser.add_argument("--num-extra-val", type=int, default=0, help="How many of the appended videos are val")
    parser.add_argument("--list-places", action="store_true", help="Print candidate videos per place and exit")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.out.read_text()) if args.out.exists() else None
    if args.extend:
        if manifest is None:
            raise SystemExit(f"--extend needs an existing manifest: {args.out}")
        if args.camera_profile != manifest["camera_profile"]:
            raise SystemExit(f"--camera-profile {args.camera_profile} != manifest's {manifest['camera_profile']}")
        if args.num_extra <= 0:
            raise SystemExit("--extend needs --num-extra > 0")
    elif manifest is not None and not args.overwrite and not args.list_places:
        print(f"Manifest exists, keeping it: {args.out}", file=sys.stderr)
        return

    root = Path(manifest["root"]) if manifest is not None and (args.extend or args.list_places) else args.root.resolve()
    annotations = read_annotations(args.csv, max_roaming_line=args.max_roaming_line)
    if args.list_places:
        list_places(annotations, root, manifest, args.place_regex)
        return

    from deepreefmap.camera.intrinsics import CameraProfile

    profile = CameraProfile.load(args.camera_profile)
    width, height = profile.image_size
    filters = dict(
        width=width,
        height=height,
        min_fps=args.min_fps,
        min_duration_s=args.min_duration_s,
        min_segment_s=args.min_segment_s,
    )

    if args.extend:
        added = extend(
            manifest,
            annotations,
            args.place_regex,
            num_extra=args.num_extra,
            num_extra_val=args.num_extra_val,
            seed=args.seed,
            **filters,
        )
        if not added:
            print(f"Nothing appended, keeping {args.out}", file=sys.stderr)
            return
        manifest.setdefault("extensions", []).append(
            {
                "csv": str(args.csv.resolve()),
                "max_roaming_line": args.max_roaming_line,
                "place_regex": args.place_regex,
                "num_extra": args.num_extra,
                "num_extra_val": args.num_extra_val,
                "num_added": len(added),
                "seed": args.seed,
            }
        )
        shutil.copyfile(args.out, args.out.with_name(f"{args.out.name}.bak"))
        write_manifest(manifest, args.out)
        print(f"Appended {len(added)} videos to {args.out} ({len(manifest['videos'])} total)", file=sys.stderr)
        return

    videos = select(
        root,
        filter_places(annotations, args.place_regex),
        num_videos=args.num_videos,
        num_val=args.num_val,
        seed=args.seed,
        **filters,
    )
    manifest = {
        "root": str(root),
        "csv": str(args.csv.resolve()),
        "max_roaming_line": args.max_roaming_line,
        "place_regex": args.place_regex,
        "camera_profile": args.camera_profile,
        "seed": args.seed,
        "videos": videos,
    }
    write_manifest(manifest, args.out)
    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
