"""Recompute chunk boundary metrics from an existing run directory.

Given a run dir that contains ``yolo_tracking.parquet``, recomputes per-frame
metrics, adaptive chunk boundaries, and saves updated outputs in-place:

- metrics/yolo_scan_metrics.parquet, metrics/yolo_scan_summary.parquet
- chunk_info.json
- visualizations/yolo_scan_overview.png
- visualizations/chunk_boundaries.png (frame screengrab grid, requires video)

Usage::

    pixi run -e tracker python -m pipeline.compute_chunk_boundaries \\
        --run-dir data/results/tracking/{config_stem}/day_{N}/{video_stem}

    pixi run -e tracker python -m pipeline.compute_chunk_boundaries \\
        --run-dir data/results/tracking/{config_stem}/day_{N}/{video_stem} \\
        --video-dir data/videos/day_{N}
"""

import json

from argparse import ArgumentParser
from pathlib import Path

import cv2
import pandas as pd
import yaml

from loguru import logger
from omegaconf import OmegaConf

from src.tracking.chunking import chunk_video_frames_adaptive
from src.tracking.metrics import compute_yolo_per_frame_metrics
from src.tracking.scan import yolo_scan_to_df
from src.tracking.viz import plot_chunk_boundary_frames, plot_yolo_scan_overview


def _resolve_video_path(
    run_dir: Path,
    video_path_override: Path | None = None,
    video_dir_override: Path | None = None,
) -> Path | None:
    """Resolve the video file from overrides, saved config, or common locations."""
    video_stem = run_dir.name

    if video_path_override is not None:
        p = video_path_override.expanduser().resolve()
        if p.exists():
            return p
        logger.warning(f"Video not found: {p}")
        return None

    if video_dir_override is not None:
        candidate = video_dir_override / f"{video_stem}.mp4"
        if candidate.exists():
            return candidate
        logger.warning(f"Video not found: {candidate}")

    config_files = list(run_dir.glob("*.yaml"))
    cfg = {}
    if config_files:
        with open(config_files[0], "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    raw = cfg.get("video_path")
    if raw is not None:
        p = Path(raw)
        if not p.is_absolute():
            p = Path.cwd() / p
        if p.exists():
            return p

    raw_dir = cfg.get("video_dir")
    if raw_dir is not None:
        vdir = Path(raw_dir)
        if not vdir.is_absolute():
            vdir = Path.cwd() / vdir
        candidate = vdir / f"{video_stem}.mp4"
        if candidate.exists():
            return candidate

    for search_dir in [
        Path.cwd() / "data" / "videos",
    ]:
        if not search_dir.exists():
            continue
        matches = list(search_dir.rglob(f"{video_stem}.mp4"))
        if matches:
            return matches[0]

    logger.warning(
        f"Could not find video for {video_stem}; use --video-path or --video-dir"
    )
    return None


def parse_args():
    parser = ArgumentParser(
        description="Recompute YOLO scan metrics and chunk boundaries from an existing run dir"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Existing run directory containing yolo_tracking.parquet",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Override YAML config; defaults to the .yaml found in --run-dir",
    )
    parser.add_argument(
        "--video-path",
        default=None,
        help="Explicit path to source video (for boundary frame grid)",
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        help="Directory containing source videos (matched by run dir name)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    assert run_dir.exists(), f"Run dir not found: {run_dir}"

    # -------------------------------------------------------------------------
    # 1. Find and load config
    # -------------------------------------------------------------------------
    if args.config:
        config_path = Path(args.config)
        assert config_path.exists(), f"Config not found: {config_path}"
    else:
        yaml_files = list(run_dir.glob("*.yaml"))
        assert yaml_files, f"No YAML config found in: {run_dir}"
        assert (
            len(yaml_files) == 1
        ), f"Expected exactly one YAML in {run_dir}, found: {yaml_files}"
        config_path = yaml_files[0]

    logger.info(f"Loading config: {config_path}")
    cfg = OmegaConf.load(config_path)
    yolo_scan_cfg = cfg.get("yolo_scan", {})

    # -------------------------------------------------------------------------
    # 2. Load raw YOLO tracking data
    # -------------------------------------------------------------------------
    yolo_parquet_path = run_dir / "yolo_tracking.parquet"
    assert yolo_parquet_path.exists(), f"yolo_tracking.parquet not found in: {run_dir}"
    logger.info(f"Loading: {yolo_parquet_path}")
    yolo_df = pd.read_parquet(yolo_parquet_path)

    # -------------------------------------------------------------------------
    # 3. Get fps + total_frames
    #    Primary: existing yolo_scan_summary.parquet  Fallback: cv2
    # -------------------------------------------------------------------------
    summary_path = run_dir / "metrics" / "yolo_scan_summary.parquet"
    fps: float
    total_frames: int

    if summary_path.exists():
        summary_df = pd.read_parquet(summary_path)
        fps = float(summary_df["fps"].iloc[0])
        total_frames = int(summary_df["total_frames"].iloc[0])
        logger.info(f"fps={fps}, total_frames={total_frames} (from summary parquet)")
    else:
        logger.warning(f"{summary_path} not found — falling back to cv2")
        video_path = cfg.get("video_path")
        assert video_path, "video_path not in config and no summary parquet available"
        cap_meta = cv2.VideoCapture(str(video_path))
        assert cap_meta.isOpened(), f"Could not open video: {video_path}"
        fps = cap_meta.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_meta.release()
        logger.info(f"fps={fps}, total_frames={total_frames} (from cv2)")

    # -------------------------------------------------------------------------
    # 4. Per-frame metrics
    # -------------------------------------------------------------------------
    occlusion_iou_threshold = float(yolo_scan_cfg.get("occlusion_iou_threshold", 0.15))
    clustering_distance_threshold = float(
        yolo_scan_cfg.get("clustering_distance_threshold", 0.15)
    )
    separation_min_objects = int(yolo_scan_cfg.get("separation_min_objects", 3))
    separation_min_distance = float(yolo_scan_cfg.get("separation_min_distance", 0.15))

    logger.info("Computing per-frame metrics...")
    per_frame_metrics = compute_yolo_per_frame_metrics(
        yolo_df,
        occlusion_iou_threshold=occlusion_iou_threshold,
        clustering_distance_threshold=clustering_distance_threshold,
        use_normalized_coords=True,
        separation_min_objects=separation_min_objects,
        separation_min_distance=separation_min_distance,
    )
    logger.info(f"  {len(per_frame_metrics)} frames with metrics")

    # -------------------------------------------------------------------------
    # 5. Adaptive chunk boundaries
    # -------------------------------------------------------------------------
    chunk_seconds = float(cfg.get("chunk_seconds", 60))
    search_window_seconds = float(cfg.get("adaptive_search_window_seconds", 10.0))
    max_chunk_seconds = float(cfg.get("adaptive_max_chunk_seconds", 150))

    logger.info("Computing adaptive chunk boundaries...")
    chunks = chunk_video_frames_adaptive(
        total_frames,
        fps,
        chunk_seconds,
        per_frame_metrics=per_frame_metrics,
        search_window_seconds=search_window_seconds,
        max_chunk_seconds=max_chunk_seconds,
    )
    logger.info(f"  {len(chunks)} chunks:")
    for i, (s, e, mtype) in enumerate(chunks):
        logger.info(f"    Chunk {i}: frames {s}–{e} ({(e - s) / fps:.1f}s) [{mtype}]")

    # -------------------------------------------------------------------------
    # 8. Save parquets in-place
    # -------------------------------------------------------------------------
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    yolo_scan_metrics_df = yolo_scan_to_df(per_frame_metrics)
    yolo_scan_metrics_path = metrics_dir / "yolo_scan_metrics.parquet"
    yolo_scan_metrics_df.to_parquet(yolo_scan_metrics_path, index=False)
    logger.info(f"Saved: {yolo_scan_metrics_path}")

    yolo_scan_summary_df = pd.DataFrame(
        [
            {
                "total_frames": total_frames,
                "video_duration_seconds": total_frames / fps,
                "fps": fps,
            }
        ]
    )
    yolo_scan_summary_path = metrics_dir / "yolo_scan_summary.parquet"
    yolo_scan_summary_df.to_parquet(yolo_scan_summary_path, index=False)
    logger.info(f"Saved: {yolo_scan_summary_path}")

    # -------------------------------------------------------------------------
    # 9. Save chunk_info.json in-place
    # -------------------------------------------------------------------------
    chunk_info = {
        "chunks": [
            {
                "chunk_idx": i,
                "frame_range": [s, e],
                "model_type": (
                    "Sam3VideoModel" if mtype == "video" else "Sam3TrackerVideoModel"
                ),
            }
            for i, (s, e, mtype) in enumerate(chunks)
        ]
    }
    chunk_info_path = run_dir / "chunk_info.json"
    with open(chunk_info_path, "w", encoding="utf-8") as f:
        json.dump(chunk_info, f, indent=2)
    logger.info(f"Saved: {chunk_info_path}")

    # -------------------------------------------------------------------------
    # 10. Regenerate yolo_scan_overview.png
    # -------------------------------------------------------------------------
    viz_dir = run_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    tracker_chunk_starts = [s for s, e, mtype in chunks[1:]]

    plot_yolo_scan_overview(
        yolo_scan_metrics_df,
        occlusion_periods=None,
        chunk_boundaries=tracker_chunk_starts,
        fps=fps,
        save_path=viz_dir / "yolo_scan_overview.png",
    )
    logger.info(f"Saved: {viz_dir / 'yolo_scan_overview.png'}")

    # -------------------------------------------------------------------------
    # 11. Generate chunk_boundaries.png (N rows × 2 cols: start + end)
    # -------------------------------------------------------------------------
    video_path = _resolve_video_path(
        run_dir,
        video_path_override=Path(args.video_path) if args.video_path else None,
        video_dir_override=Path(args.video_dir) if args.video_dir else None,
    )
    if video_path is not None:
        boundary_path = viz_dir / "chunk_boundaries.png"
        plot_chunk_boundary_frames(
            chunk_info=chunk_info,
            video_path=video_path,
            fps=fps,
            yolo_scan_df=yolo_scan_metrics_df,
            save_path=boundary_path,
        )
        logger.info(f"Saved: {boundary_path}")
    else:
        logger.warning("Skipping chunk_boundaries.png (no video found)")


if __name__ == "__main__":
    main()
