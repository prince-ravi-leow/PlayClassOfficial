"""Barebones SAM 3 tracker: fixed-length chunks, each seeded by text grounding.

For every fixed chunk (default 60 s):

1. Run text-prompted ``Sam3VideoModel`` on the first ``--grounding-frames``
   frames and pick the cleanest one (``find_best_grounding_frame``).
2. Turn each object's mask into 3 click points and run point-prompted
   ``Sam3TrackerVideoModel`` from that frame to the end of the chunk.
3. Frames before the chosen frame keep the grounding model's own outputs.

If no grounding frame has ``--min-objects`` objects, the chunk is seeded from
the last frames of the previous chunk instead, or left empty if there is none.

Object IDs are only consistent *within* a chunk; the ``chunk_idx`` column in
the output tells chunks apart.

Outputs, in ``{out_dir}/{video_stem}/``:

- tracking_outputs.parquet (same schema as ``pipeline/run_tracker.py``)
- annotated_video.mp4 (supervision masks, boxes and IDs)
- sam3_simple_{timestamp}.log

Usage::

    pixi run -e tracker python -m pipeline.run_tracker_simple \\
        --video-path path/to/video.mp4 --text-prompt bird
"""

import os

os.environ.setdefault(
    "PYTORCH_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.6"
)

from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv

from accelerate import Accelerator
from loguru import logger
from omegaconf import OmegaConf

from src._config import DEFAULT_TRACKING_DIR
from src.tracking.chunking import chunk_video_frames_adaptive
from src.tracking.grounding import find_best_grounding_frame, run_grounding
from src.tracking.sam3 import _process_tracker_chunk, _process_video_chunk
from src.tracking.utils import (
    extract_equidistant_points_from_masks,
    find_frame_with_enough_objects,
    process_tracking_outputs,
)
from src.tracking.viz import create_annotation_callback
from src.utils.io import (
    get_video_metadata,
    load_video_frames,
    sanitize_filename,
    setup_logger,
)
from src.utils.memory import free_gpu_memory

# SAM 3 tracker settings, copied from config/sam3_best.yaml
TRACKING_PARAMS = {
    "init_trk_keep_alive": 60,
    "max_trk_keep_alive": 60,
    "min_trk_keep_alive": 0,
    "trk_assoc_iou_thresh": 0.3,
    "hotstart_dup_thresh": 12,
    "suppress_overlap_thresh": 0.8,
    "recondition_every_nth_frame": 8,
}
# Fallback seeding: how many of the previous chunk's last frames to search
MAX_LOOKBACK_FRAMES = 10


def _prompt_points(masks: list, object_ids: list) -> dict[int, list]:
    """Three click points per object, keyed by object ID. Empty masks are skipped."""
    masks_arr = np.stack(
        [m.squeeze(0) if m.ndim == 3 and m.shape[0] == 1 else m for m in masks]
    ).astype(bool)
    points = extract_equidistant_points_from_masks(masks_arr, num_points=3)
    return {
        int(oid): points[i].tolist()
        for i, oid in enumerate(object_ids)
        if masks_arr[i].any()
    }


def track_video(
    video_path: Path,
    run_dir: Path,
    text_prompt: str,
    chunk_seconds: float,
    grounding_frames: int,
    min_objects: int,
) -> None:
    """Track one video chunk by chunk, writing the parquet and annotated video."""
    # The _process_*_chunk helpers read their settings from an OmegaConf config
    cfg = OmegaConf.create({"text_prompt": text_prompt, "tracking": TRACKING_PARAMS})
    device = Accelerator().device

    fps, total_frames = get_video_metadata(video_path)
    if fps <= 0 or total_frames <= 0:
        raise ValueError(
            f"Could not read FPS/frame count from {video_path} "
            f"(fps={fps}, frames={total_frames})"
        )
    # Without YOLO metrics this returns plain fixed-length chunks
    # (a tail shorter than 10% of a chunk is merged into the last one)
    chunks = chunk_video_frames_adaptive(total_frames, fps, chunk_seconds)
    logger.info(
        f"{video_path.name}: {total_frames} frames at {fps:.2f} FPS -> "
        f"{len(chunks)} chunk(s) of {chunk_seconds:g}s, "
        f"grounding on the first {grounding_frames} frames of each"
    )

    results_path = run_dir / "tracking_outputs.parquet"
    video_out_path = run_dir / "annotated_video.mp4"
    chunk_dfs = []
    prev_tail = {}  # last frames of the previous chunk, for fallback seeding
    video_info = sv.VideoInfo.from_video_path(str(video_path))

    with sv.VideoSink(str(video_out_path), video_info) as sink:
        for chunk_idx, (start, end, _) in enumerate(chunks):
            logger.info(f"Chunk {chunk_idx}/{len(chunks) - 1}: frames {start}-{end}")
            frames = load_video_frames(
                video_path, start, end, loader="torchcodec", device=str(device)
            )

            # 1. Ground: text-prompted SAM 3 on the first frames, keep the cleanest
            grounding_outputs = run_grounding(
                frames, start, grounding_frames, _process_video_chunk, cfg, device
            )
            seed_idx, masks, _, object_ids = find_best_grounding_frame(
                grounding_outputs, min_objects=min_objects
            )
            prefix = {}
            if seed_idx is not None:
                # Frames before the seed frame keep the grounding outputs. The
                # tracker reuses the grounding IDs, so no remapping is needed.
                prefix = {
                    i: grounding_outputs[i]
                    for i in range(start, seed_idx)
                    if i in grounding_outputs
                }
                logger.info(
                    f"  seeded from grounding frame {seed_idx} "
                    f"({len(object_ids)} objects)"
                )
            else:
                # 2. Fallback: seed from the previous chunk's last good frame
                src_idx, masks, _, object_ids = find_frame_with_enough_objects(
                    prev_tail, min_objects=min_objects, max_lookback=MAX_LOOKBACK_FRAMES
                )
                if src_idx is not None:
                    seed_idx = start
                    logger.warning(
                        f"  grounding found no frame with >= {min_objects} objects; "
                        f"seeded from previous-chunk frame {src_idx}"
                    )
            del grounding_outputs

            # 3. Track from the seed frame to the end of the chunk
            points = _prompt_points(masks, object_ids) if masks else {}
            if points:
                tracked = _process_tracker_chunk(
                    frames[seed_idx - start :], seed_idx, points, cfg, device
                )
            else:
                tracked = {}
                logger.warning(f"  no objects to track; chunk {chunk_idx} left empty")
            free_gpu_memory()

            for out in prefix.values():
                out["_model_type"] = "Sam3VideoModel"
            for out in tracked.values():
                out["_model_type"] = "Sam3TrackerVideoModel"
            chunk_outputs = {**prefix, **tracked}
            for frame_idx, out in chunk_outputs.items():
                out["_chunk_idx"] = chunk_idx
                out["_is_chunk_start"] = frame_idx == start

            # 4. Save (rewritten every chunk, so a crash keeps finished chunks)
            if chunk_outputs:
                chunk_dfs.append(process_tracking_outputs(chunk_outputs).sort_index())
                pd.concat(chunk_dfs).to_parquet(results_path)

            # 5. Annotate this chunk's frames while they are still in RAM
            annotate = create_annotation_callback(chunk_outputs)
            for i, frame in enumerate(frames):
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                sink.write_frame(annotate(bgr, start + i))

            prev_tail = {
                i: chunk_outputs[i] for i in sorted(chunk_outputs)[-MAX_LOOKBACK_FRAMES:]
            }
            # Free this chunk before the next one is loaded
            del frames, annotate, chunk_outputs, tracked, prefix

    if chunk_dfs:
        logger.info(f"Tracking results saved to: {results_path}")
    else:
        logger.warning("No objects tracked in any chunk; no parquet written")
    logger.info(f"Annotated video saved to: {video_out_path}")


def main():
    parser = ArgumentParser(
        description="Barebones SAM 3 tracker: fixed chunks seeded by text grounding."
    )
    parser.add_argument(
        "--video-path", type=Path, required=True, help="Input video file."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(DEFAULT_TRACKING_DIR) / "simple",
        help="Output root; results go in {out_dir}/{video_stem}/.",
    )
    parser.add_argument(
        "--text-prompt", default="bird", help="What SAM 3 should segment."
    )
    parser.add_argument(
        "--chunk-seconds", type=float, default=60.0, help="Chunk length in seconds."
    )
    parser.add_argument(
        "--grounding-frames",
        type=int,
        default=125,
        help="Frames at the start of each chunk scanned for the best seed frame.",
    )
    parser.add_argument(
        "--min-objects",
        type=int,
        default=3,
        help="Objects a grounding frame needs before it can seed the tracker.",
    )
    args = parser.parse_args()

    if not args.video_path.is_file():
        raise FileNotFoundError(f"Video not found: {args.video_path}")

    run_dir = args.out_dir / sanitize_filename(args.video_path.stem)
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(run_dir, job_type="sam3_simple")

    track_video(
        video_path=args.video_path,
        run_dir=run_dir,
        text_prompt=args.text_prompt,
        chunk_seconds=args.chunk_seconds,
        grounding_frames=args.grounding_frames,
        min_objects=args.min_objects,
    )


if __name__ == "__main__":
    main()
