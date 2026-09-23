"""Build the behaviour dataset from tracking outputs and postprocessing JSONs.

Reads ``tracking_outputs.parquet`` from ``--tracking-dir`` and reads/writes postprocessing JSONs
(``tracking_issues.json``, ``tracking_postprocessing.json``, ``bird_info.json``)
under ``--postprocessing-dir``.

Steps:

1. Discovers all tracking subdirectories under ``--tracking-dir``
2. Parses labels and bird info from Excel files (once)
3. For each video:
   a. Reads ``tracking_outputs.parquet`` + FPS from ``--tracking-dir``
   b. Checks ``--postprocessing-dir`` for existing JSONs
   c. If missing: detects issues, prefills postprocessing, writes JSONs
   d. If present: loads ``tracking_postprocessing.json``
4. Once all remaps are filled in, runs ``process_tracks`` and concatenates.

Typical workflow::

    # First run: generates JSONs in data/postprocessing/
    pixi run python -m pipeline.build_dataset

    # Manually fill in "to" values in each tracking_postprocessing.json

    # Second run: validates remaps, builds dataset
    pixi run python -m pipeline.build_dataset
"""

import json
import os

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from glob import glob
from pathlib import Path

import pandas as pd

from loguru import logger

from src._config import (
    DEFAULT_DATASET_DIR,
    DEFAULT_LABEL_DIR,
    DEFAULT_MIN_WINDOW_COVERAGE,
    DEFAULT_POSTPROCESSING_DIR,
    DEFAULT_TRACKING_DIR,
)
from src.dataset.labels import process_labels, resolve_dual_groups
from src.dataset.tracking_issues import detect_tracking_issues
from src.dataset.tracking_postprocessing import (
    align_labels,
    assign_windows,
    filter_incomplete_windows,
    prefill_postprocessing,
    process_tracks,
)
from src.dataset.utils import extract_video_id, fmt_time, get_video_fps


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _log_issues(issues, fps):
    """Log a summary of detected tracking issues."""
    n_switches = sum(1 for v in issues.values() if v["type"] == "id_switch")
    n_overlaps = sum(1 for v in issues.values() if v["type"] == "overlap")
    n_low = sum(1 for v in issues.values() if v["type"] == "low_score")
    logger.info(
        f"  Found {n_switches} ID switch(es), {n_overlaps} overlap(s), "
        f"{n_low} low-score period(s)"
    )

    for name, ev in issues.items():
        if ev["type"] == "overlap":
            dur = ev["end_time"] - ev["start_time"]
            logger.info(
                f"    {name}: IDs {ev['ids']}  "
                f"{fmt_time(ev['start'], fps)} - {fmt_time(ev['end'], fps)}  "
                f"({dur:.1f}s)"
            )
        elif ev["type"] == "low_score":
            dur = ev["end_time"] - ev["start_time"]
            logger.info(
                f"    {name}: ID {ev['id']}  "
                f"{fmt_time(ev['start'], fps)} - {fmt_time(ev['end'], fps)}  "
                f"({dur:.1f}s, min_score={ev['min_score']:.3f}, "
                f"{ev['low_score_rows']}/{ev['total_rows']} rows)"
            )
        else:
            logger.info(
                f"    {name}: frame {ev['start']}  "
                f"({fmt_time(ev['start'], fps)})  "
                f"{ev['ids_before']} -> {ev['ids_after']}"
            )


def process_tracking_subdir(tracking_dir, pp_dir, bird_info):
    """Detect issues, write JSONs to pp_dir, return in-memory results.

    Reads parquets from *tracking_dir*, reads/writes postprocessing JSONs
    from/to *pp_dir*.  If ``tracking_postprocessing.json`` already exists in
    *pp_dir*, skips issue detection and bird_info generation.

    Returns
    -------
    dict
        ``tracks``: DataFrame from tracking_outputs.parquet
        ``postprocessing``: list of postprocessing entries (trims + remaps)
        ``video_id``: extracted video ID (e.g. ``C1G3D28``)
        ``fps``: video FPS
    """
    tracking_dir = Path(tracking_dir)
    pp_dir = Path(pp_dir)
    pp_dir.mkdir(parents=True, exist_ok=True)
    video_id = extract_video_id(tracking_dir.name)

    fps = get_video_fps(tracking_dir)
    tracks = pd.read_parquet(tracking_dir / "tracking_outputs.parquet").reset_index()
    tracks = tracks.rename(columns={"object_id": "tracking_id"})

    pp_path = pp_dir / "tracking_postprocessing.json"
    if pp_path.exists():
        postprocessing = _load_json(pp_path)
        logger.info(
            f"Loading tracking data for {tracking_dir.name} "
            f"({len(postprocessing)} postprocessing entry/ies, {len(tracks)} rows)"
        )
    else:
        logger.info(f"Loading tracking data for {tracking_dir.name} (new)")
        logger.info(f"  FPS: {fps:.2f}, {len(tracks)} tracking rows")

        issues = detect_tracking_issues(tracks, fps)
        _log_issues(issues, fps)
        _save_json(pp_dir / "tracking_issues.json", issues)

        video_birds = bird_info.get(video_id, {})
        if video_birds:
            _save_json(pp_dir / "bird_info.json", video_birds)
            logger.info(f"  Saved bird_info.json ({len(video_birds)} bird(s))")
        else:
            logger.warning(f"  No bird info matched for {tracking_dir.name}")

        postprocessing = prefill_postprocessing(issues, tracks, video_birds, fps)
        _save_json(pp_path, postprocessing)
        logger.info(
            f"  Created tracking_postprocessing.json with "
            f"{len(postprocessing)} template(s)"
        )

    return {
        "tracks": tracks,
        "postprocessing": postprocessing,
        "video_id": video_id,
        "fps": fps,
    }


def save_data(save_dict, output_dir):
    """Save a dict of DataFrames/JSON to an output directory (parquet or JSON by extension)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fname, data in save_dict.items():
        path = output_dir / fname
        if path.suffix == ".json":
            _save_json(path, data)
        elif path.suffix == ".parquet":
            data.to_parquet(path)
        else:
            raise ValueError(f"Unsupported file type for {fname}, skipping save")
        logger.info(f"Saved: {path}")


def parse_args():
    parser = ArgumentParser(
        description="Build tracking_issues.json per subdirectory, then build dataset.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tracking-dir",
        type=Path,
        default=f"{DEFAULT_TRACKING_DIR}/sam3_best",
        help="Root dir to search for tracking_outputs.parquet.",
    )
    parser.add_argument(
        "--postprocessing-dir",
        type=Path,
        default=DEFAULT_POSTPROCESSING_DIR,
        help="Dir to read/write postprocessing JSONs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory to write dataset outputs.",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=DEFAULT_LABEL_DIR,
        help="Path to directory containing Registration protocols Excel files",
    )
    parser.add_argument(
        "--tracking-fname",
        type=str,
        default="tracking_outputs.parquet",
        help="Filename to look for in each tracking subdirectory",
    )
    parser.add_argument(
        "--min-window-coverage",
        type=float,
        default=DEFAULT_MIN_WINDOW_COVERAGE,
        help="Minimum fraction of expected frames for a window to be kept.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Glob label files
    label_files = sorted(glob(str(args.label_dir / "*.xlsx")))
    if not label_files:
        logger.error(f"No .xlsx files found in {args.label_dir}")
        return
    logger.info(f"Found {len(label_files)} label file(s)")

    # 2. Parse labels + bird info
    labels, bird_info = process_labels(label_files)
    labels = resolve_dual_groups(labels)

    # 3. Discover tracking subdirs
    tracking_dirs = sorted(
        [
            p.parent
            for p in args.tracking_dir.rglob(f"{args.tracking_fname}")
            if p.is_file()
        ]
    )
    # rglob doesn't follow symlinks by default; fall back to glob via os.walk
    if not tracking_dirs:
        for root, _dirs, files in os.walk(args.tracking_dir, followlinks=True):
            if args.tracking_fname in files:
                tracking_dirs.append(Path(root))
        tracking_dirs.sort()
    if not tracking_dirs:
        logger.error(f"No {args.tracking_fname} found under {args.tracking_dir}")
        return
    logger.info(f"Found {len(tracking_dirs)} tracking subdirectory(ies)")

    # 4. Detect issues + prefill remaps per subdir (keeps data in memory)
    #    Map each tracking subdir to its postprocessing counterpart by
    #    preserving the relative path (e.g. day_28/{stem}/).
    results = []
    for d in tracking_dirs:
        rel = d.relative_to(args.tracking_dir)
        pp_dir = args.postprocessing_dir / rel
        results.append(process_tracking_subdir(d, pp_dir, bird_info))

    # 5. Build dataset from in-memory data
    logger.info("Building dataset...")
    all_tracks = []
    fps_lookup = {}
    for r in results:
        logger.info(f"--- {r['video_id']} ---")
        try:
            tracks_clean, labels = process_tracks(
                tracks=r["tracks"],
                labels=labels,
                postprocessing=r["postprocessing"],
                fps=r["fps"],
                video_id=r["video_id"],
            )
        except ValueError as e:
            logger.error(f"Cannot process tracks for {r['video_id']}: {e}")
            return

        tracks_clean["video_id"] = r["video_id"]
        fps_lookup[r["video_id"]] = r["fps"]

        # Validate bird count matches expected from bird_info
        actual_ids = sorted(tracks_clean["bird_id"].unique())
        expected_ids = sorted(bird_info.get(r["video_id"], {}).keys())
        if expected_ids and actual_ids != expected_ids:
            logger.warning(
                f"  {r['video_id']}: bird_id mismatch after postprocessing — "
                f"expected {expected_ids}, got {actual_ids}"
            )

        all_tracks.append(tracks_clean)

    tracks = pd.concat(all_tracks)

    # Canonical column order: video_id, bird_id, frame_idx, then the rest
    leading = ["video_id", "bird_id", "frame_idx"]
    tracks = tracks[leading + [c for c in tracks.columns if c not in leading]]

    # 5b. Align labels to track coverage per (video_id, bird_id)
    labels = align_labels(tracks, labels, fps_lookup)

    # 5c. Assign temporal windows
    tracks, labels = assign_windows(tracks, labels, fps_lookup)

    # 5d. Filter incomplete windows
    tracks, labels = filter_incomplete_windows(
        tracks, labels, fps_lookup, min_coverage=args.min_window_coverage
    )

    # Check that tracks and labels cover the same (video_id, bird_id, window) keys
    _key_cols = ["video_id", "bird_id", "window"]
    track_keys = set(tracks[_key_cols].itertuples(index=False, name=None))
    label_keys = set(labels[_key_cols].itertuples(index=False, name=None))
    assert track_keys == label_keys, (
        f"Window key mismatch: {len(track_keys - label_keys)} in tracks only, "
        f"{len(label_keys - track_keys)} in labels only"
    )

    logger.info(f"Dataset built: {len(tracks)} track rows, {len(labels)} label rows")

    # 6. Save dataset
    output_data = {
        "tracks.parquet": tracks,
        "labels.parquet": labels,
    }
    save_data(output_data, args.output_dir)


if __name__ == "__main__":
    main()
