"""Draw tracking outputs onto their source video.

Reads a tracking parquet from ``pipeline/run_tracker_simple.py`` and its video,
and writes the video with each object's mask, box and ID drawn on. The video
is written at the parquet's resolution (masks are at the tracker's, possibly
downscaled, resolution). Frames are decoded on the CPU, since torchcodec's
GPU decoding turns our HEVC footage magenta.

Runs made with ``--frame-stride N`` have only every N-th frame tracked. By
default only those frames are drawn, at the source frame rate / N (so the
video still plays in real time). With ``--all-frames`` every source frame is
drawn at the source frame rate, the frames in between showing the last
tracked frame's annotations.

Output: ``{run_id}_annotated_video.mp4`` next to the parquet, unless
``--output`` is given.

Usage::

    pixi run -e tracker python -m pipeline.draw_annotations_simple \\
        --video-path path/to/video.mp4 \\
        --tracking-parquet tmp/{video_stem}/{run_id}_tracking_outputs.parquet
"""

import multiprocessing
import os
import subprocess
import tempfile
import time

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import supervision as sv
import torch

from loguru import logger

from src.utils.io import (
    get_run_id,
    get_torchcodec_decoder,
    load_transforms,
    load_video_frames,
    setup_logger,
)

# Frames decoded at a time
DECODE_BATCH_FRAMES = 250
# Parallel drawing processes (each decodes its own segment of the video)
DEFAULT_WORKERS = 16
# Colours by object ID, as supervision's ColorLookup.TRACK picks them
PALETTE = sv.ColorPalette.DEFAULT
MASK_OPACITY = 0.5


def default_output_path(parquet_path: Path) -> Path:
    """``{run_id}_annotated_video.mp4``, next to the run's tracking parquet."""
    name = parquet_path.name.replace("_tracking_outputs.parquet", "")
    return parquet_path.with_name(f"{name}_annotated_video.mp4")


def _frame_detections(rows: pd.DataFrame) -> tuple[sv.Detections, list[str], list]:
    """Boxes, labels and mask crops for one frame's parquet rows.

    Empty masks are skipped. Each mask is kept only within its box, as
    (box slices, crop, colour): drawing those is much faster than blending
    full-frame masks.
    """
    boxes = np.stack(rows["bbox"].to_numpy()).astype(np.float32)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    rows, boxes = rows[keep], boxes[keep]
    if len(rows) == 0:
        return sv.Detections.empty(), [], []
    object_ids = rows.index.get_level_values("object_id").to_numpy().astype(int)
    scores = rows["scores"].to_numpy().astype(np.float32)
    rles = [
        {"size": list(size), "counts": counts.encode()}
        for counts, size in zip(rows["counts"], rows["size"])
    ]
    masks = mask_util.decode(rles)  # (H, W, N)
    crops = []
    for i, (x1, y1, x2, y2) in enumerate(boxes.astype(int)):
        ys, xs = slice(y1, y2 + 1), slice(x1, x2 + 1)
        color = np.array(PALETTE.by_idx(object_ids[i]).as_bgr(), dtype=np.float32)
        crops.append((ys, xs, masks[ys, xs, i].astype(bool), color))
    detections = sv.Detections(xyxy=boxes, confidence=scores, tracker_id=object_ids)
    labels = [f"#{oid} {score:.2f}" for oid, score in zip(object_ids, scores)]
    return detections, labels, crops


def _draw_masks(bgr: np.ndarray, crops: list) -> None:
    """Blend each mask's colour into ``bgr`` in place, within its box only."""
    for ys, xs, mask, color in crops:
        region = bgr[ys, xs]
        region[mask] = region[mask] * (1 - MASK_OPACITY) + color * MASK_OPACITY


def _draw_segment(
    video_path: Path,
    tracks: pd.DataFrame,
    frame_range: tuple[int, int],
    stride: int,
    write_stride: int,
    size: tuple[int, int],
    fps: float,
    out_path: Path,
    threads: int,
) -> int:
    """Draw ``tracks`` onto video frames [start, end), writing ``out_path``.

    ``start`` must be a multiple of ``stride``. Returns the frames written.
    """
    torch.set_num_threads(threads)
    height, width = size
    meta = get_torchcodec_decoder(video_path, device="cpu").metadata
    transform = None
    if (meta.height, meta.width) != (height, width):
        transform = load_transforms((height, width))
    rows_by_frame = dict(tuple(tracks.groupby(level="frame_idx")))
    lookup = sv.ColorLookup.TRACK
    box_annotator = sv.BoxAnnotator(PALETTE, thickness=2, color_lookup=lookup)
    label_annotator = sv.LabelAnnotator(PALETTE, color_lookup=lookup)
    start, end = frame_range
    video_info = sv.VideoInfo(width=width, height=height, fps=fps)

    n_written = 0
    detections, labels, crops, last_tracked = None, [], [], None
    with sv.VideoSink(str(out_path), video_info) as sink:
        for batch_start in range(start, end, DECODE_BATCH_FRAMES):
            frames = load_video_frames(
                video_path,
                batch_start,
                min(batch_start + DECODE_BATCH_FRAMES, end),
                loader="torchcodec",
                device="cpu",
                transform=transform,
            )
            for frame_idx, frame in enumerate(frames, start=batch_start):
                if frame_idx % write_stride:
                    continue
                if frame_idx in rows_by_frame:
                    detections, labels, crops = _frame_detections(
                        rows_by_frame[frame_idx]
                    )
                    last_tracked = frame_idx
                elif last_tracked is None or frame_idx - last_tracked >= stride:
                    # Not tracked, nor right after a tracked frame
                    detections, labels, crops = None, [], []
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if detections is not None and len(detections):
                    _draw_masks(bgr, crops)
                    bgr = box_annotator.annotate(bgr, detections)
                    bgr = label_annotator.annotate(bgr, detections, labels=labels)
                sink.write_frame(bgr)
                n_written += 1
    return n_written


def draw_annotations(
    video_path: Path,
    parquet_path: Path,
    out_path: Path,
    all_frames: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> None:
    """Write ``video_path`` with the tracks in ``parquet_path`` drawn on it.

    Covers the video from its first frame to the last tracked one: only the
    tracked frames (every ``stride``-th), or with ``all_frames`` every frame.
    The video is split into ``workers`` segments, drawn in parallel processes
    (each decoding its own segment) and joined with ffmpeg without re-encoding.
    """
    t_start = time.time()
    tracks = pd.read_parquet(parquet_path, columns=["bbox", "counts", "size", "scores"])
    tracked_frames = np.sort(tracks.index.get_level_values("frame_idx").unique())
    # Most common gap between tracked frames: the run's --frame-stride
    gaps = np.diff(tracked_frames)
    stride = int(np.bincount(gaps).argmax()) if len(gaps) else 1
    size = tuple(int(x) for x in tracks["size"].iloc[0])  # (H, W)
    meta = get_torchcodec_decoder(video_path, device="cpu").metadata
    end = min(meta.num_frames, int(tracked_frames[-1]) + stride)
    # Frames written: every one, or only those on the tracked (stride) grid
    write_stride = 1 if all_frames else stride
    fps = meta.average_fps / write_stride
    logger.info(
        f"Drawing {len(tracked_frames)} tracked frames (stride {stride}) onto "
        f"{'all' if all_frames else 'the tracked'} frames 0-{end} of "
        f"{video_path.name}, at {size[1]}x{size[0]} and {fps:g} fps, "
        f"with {workers} worker(s)"
    )

    # Segment starts are multiples of the stride, so each starts on a tracked frame
    step = -(-end // workers // stride) * stride
    ranges = [(a, min(a + step, end)) for a in range(0, end, step)]
    frame_idx = tracks.index.get_level_values("frame_idx")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    threads = max(1, (os.cpu_count() or 1) // len(ranges))
    with tempfile.TemporaryDirectory(dir=out_path.parent) as tmp_dir:
        parts = [Path(tmp_dir) / f"part{i:03d}.mp4" for i in range(len(ranges))]
        jobs = [
            (
                video_path,
                tracks[(frame_idx >= a) & (frame_idx < b)],
                (a, b),
                stride,
                write_stride,
                size,
                fps,
                part,
                threads,
            )
            for (a, b), part in zip(ranges, parts)
        ]
        # Spawned, not forked: decoders opened by this process must not be shared
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(len(jobs), mp_context=context) as pool:
            futures = [pool.submit(_draw_segment, *job) for job in jobs]
            n_written = 0
            for done, future in enumerate(as_completed(futures), start=1):
                n_written += future.result()
                logger.info(f"  {done}/{len(jobs)} segments drawn")
        concat_list = Path(tmp_dir) / "parts.txt"
        # ffmpeg resolves these relative to the list file's directory
        concat_list.write_text("".join(f"file '{part.name}'\n" for part in parts))
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0"]
            + ["-i", str(concat_list), "-c", "copy", str(out_path)],
            check=True,
        )
    logger.info(
        f"Annotated video saved to: {out_path} "
        f"({n_written} frames in {time.time() - t_start:.0f}s)"
    )


def main():
    parser = ArgumentParser(
        description="Draw tracking outputs (parquet) onto their source video.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video-path", type=Path, required=True, help="Source video file."
    )
    parser.add_argument(
        "--tracking-parquet",
        type=Path,
        required=True,
        help="{run_id}_tracking_outputs.parquet from run_tracker_simple.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video; default: {run_id}_annotated_video.mp4 next to the parquet.",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="For --frame-stride runs, draw every source frame, not just tracked ones.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel drawing processes, each on its own segment of the video.",
    )
    args = parser.parse_args()
    for path in (args.video_path, args.tracking_parquet):
        if not path.is_file():
            raise FileNotFoundError(f"Not found: {path}")
    out_path = args.output or default_output_path(args.tracking_parquet)

    run_id = get_run_id()
    log_path = setup_logger(
        out_path.parent, job_type="draw_annotations_simple", run_id=run_id
    )
    logger.info(f"Run id: {run_id}, logging to: {log_path}")
    logger.info(f"Arguments: {vars(args)}")

    draw_annotations(
        args.video_path,
        args.tracking_parquet,
        out_path,
        all_frames=args.all_frames,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
