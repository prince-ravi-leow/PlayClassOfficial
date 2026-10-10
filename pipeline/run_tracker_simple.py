"""Barebones SAM 3 tracker: fixed-length chunks, each seeded by text grounding.

For every fixed chunk (default 60 s):

1. Run a text-prompted model (``--grounder``) on the first
   ``--grounding-frames`` frames and pick the one with the most objects
   (ties: least overlap, then highest score; ``find_best_grounding_frame``):

   - ``sam3`` (default): ``Sam3VideoModel`` via transformers
   - ``sam3.1``: the SAM 3.1 video predictor (``ext/sam3`` submodule);
     ``--compile`` compiles the SAM 3.1 models (slow first run)
2. Hand each object to a promptable tracker (``--tracker``), as its mask
   (default) or as 3 click points sampled inside it (``--prompt points``),
   and track from that frame to the end of the chunk:

   - ``sam3`` (default): ``Sam3TrackerVideoModel``
   - ``sam2``: SAM 2.1 Large via transformers (``Sam2VideoModel``)
   - ``sam2-hq``: HQ-SAM 2 Large (SysCV/sam-hq); fetch its code first with
     ``pixi run -e tracker setup_hqsam2``
   - ``sam3.1``: the SAM 3.1 multiplex tracker (``ext/sam3`` submodule); it
     also tracks backwards from the chosen frame to the chunk start
3. Frames before the chosen frame keep the grounding model's own outputs
   (except with ``--tracker sam3.1``, which covers them).

If no grounding frame has ``--min-objects`` objects, the chunk is seeded from
the last frames of the previous chunk instead, or left empty if there is none.

Object IDs are only consistent *within* a chunk; the ``chunk_idx`` column in
the output tells chunks apart.

With ``--frame-stride N`` only every N-th frame is tracked (e.g. 2: 25 -> 12.5
fps), so the parquet only has those frames; frame numbers stay those of the
original video.

Frames whose short side is over ``--max-short-side`` pixels (default 1080) are
downscaled while decoding (e.g. 4K -> 1080p), so all outputs, including mask
and box coordinates, are at the downscaled resolution.

Outputs, in ``{out_dir}/{video_stem}/``, prefixed with a unique run id
(timestamp, e.g. ``2026-10-06_18-28-51.37``):

- {run_id}_tracking_outputs.parquet (same schema as ``pipeline/run_tracker.py``)
- {run_id}_annotated_video.mp4 (masks, boxes and IDs drawn on the source video
  by ``pipeline/draw_annotations_simple.py``; skip with ``--no-annotated-video``)
- {run_id}_{grounder}-{tracker}_simple.log

Usage::

    pixi run -e tracker python -m pipeline.run_tracker_simple \\
        --video-path path/to/video.mp4 --text-prompt bird --tracker sam2

SAM 3.1 needs the submodule (``git submodule update --init ext/sam3``) and
access to the gated ``facebook/sam3.1`` Hugging Face repo.
"""

import math
import os
import time

os.environ.setdefault(
    "PYTORCH_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.6"
)

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from loguru import logger
from omegaconf import OmegaConf

from pipeline.draw_annotations_simple import draw_annotations
from src._config import DEFAULT_TRACKING_DIR
from src.tracking.chunking import chunk_video_frames_adaptive
from src.tracking.grounding import find_best_grounding_frame, run_grounding
from src.tracking.sam2_hq import track_chunk_sam2_hq
from src.tracking.sam3 import _process_tracker_chunk, _process_video_chunk
from src.tracking.sam3_1 import Sam31
from src.tracking.utils import (
    extract_equidistant_points_from_masks,
    find_frame_with_enough_objects,
    process_tracking_outputs,
)
from src.utils.io import (
    get_downscaled_size,
    get_run_id,
    get_torchcodec_decoder,
    load_transforms,
    load_video_frames,
    sanitize_filename,
    setup_logger,
)
from src.utils.memory import free_gpu_memory

# SAM 3 tracker settings, copied from config/sam3_best.yaml. Disabled for now:
# the models run with their facebook/sam3 defaults.
# TRACKING_PARAMS = {
#     "init_trk_keep_alive": 60,
#     "max_trk_keep_alive": 60,
#     "min_trk_keep_alive": 0,
#     "trk_assoc_iou_thresh": 0.3,
#     "hotstart_dup_thresh": 12,
#     "suppress_overlap_thresh": 0.8,
#     "recondition_every_nth_frame": 8,
# }
# Fallback seeding: how many of the previous chunk's last frames to search
MAX_LOOKBACK_FRAMES = 10
# Decode on the CPU: torchcodec's GPU (NVDEC) decoding turns our HEVC
# yuvj420p footage magenta, which is what the models would then see
DECODE_DEVICE = "cpu"
# --grounder / --tracker choice -> model name in the parquet's model_type column
GROUNDER_MODEL_TYPES = {
    "sam3": "Sam3VideoModel",
    "sam3.1": "Sam3MultiplexVideoPredictor",
}
TRACKER_MODEL_TYPES = {
    "sam3": "Sam3TrackerVideoModel",
    "sam2": "Sam2VideoModel",
    "sam2-hq": "SAM2HQVideoPredictor",
    "sam3.1": "Sam3VideoTrackingMultiplex",
}


def _seed_prompts(
    masks: list, object_ids: list
) -> tuple[dict[int, list], dict[int, np.ndarray]]:
    """Three click points and the (H, W) mask per object, both keyed by object ID.

    Empty masks are skipped.
    """
    masks_arr = np.stack([
        m.squeeze(0) if m.ndim == 3 and m.shape[0] == 1 else m for m in masks
    ]).astype(bool)
    points = extract_equidistant_points_from_masks(masks_arr, num_points=3)
    keep = [i for i in range(len(object_ids)) if masks_arr[i].any()]
    return (
        {int(object_ids[i]): points[i].tolist() for i in keep},
        {int(object_ids[i]): masks_arr[i] for i in keep},
    )


def track_video(
    video_path: Path,
    run_dir: Path,
    run_id: str,
    text_prompt: str,
    chunk_seconds: float,
    grounding_frames: int,
    min_objects: int,
    max_short_side: int,
    max_frames_to_track: int = 0,
    tracker: str = "sam3",
    prompt: str = "masks",
    grounder: str = "sam3",
    compile: bool = False,
    frame_stride: int = 1,
    annotate: bool = True,
) -> None:
    """Track one video chunk by chunk, writing the parquet and annotated video.

    ``max_frames_to_track`` > 0 stops after that many frames; 0 tracks them all.
    ``grounder`` and ``tracker`` are keys of ``GROUNDER_MODEL_TYPES`` and
    ``TRACKER_MODEL_TYPES``; ``prompt`` is "masks" or "points". ``compile``
    applies ``torch.compile`` to the SAM 3.1 models. ``frame_stride`` > 1
    tracks only every ``frame_stride``-th frame. ``annotate`` draws the tracks
    onto the source video afterwards.
    """
    t_start = time.time()
    # The _process_*_chunk helpers read their settings from an OmegaConf config.
    # No "tracking" key, so they keep the facebook/sam3 defaults.
    cfg = OmegaConf.create({"text_prompt": text_prompt})
    device = Accelerator().device
    # SAM 3.1 models are built once, on first use, and kept for all chunks
    sam31 = Sam31(device, compile=compile) if "sam3.1" in (grounder, tracker) else None
    ground_chunk = sam31.ground if grounder == "sam3.1" else _process_video_chunk

    # One torchcodec decoder supplies both the metadata and (later) the frames
    meta = get_torchcodec_decoder(video_path, device=DECODE_DEVICE).metadata
    fps, total_frames = meta.average_fps, meta.num_frames
    if fps <= 0 or total_frames <= 0:
        raise ValueError(
            f"Could not read FPS/frame count from {video_path} "
            f"(fps={fps}, frames={total_frames})"
        )
    if 0 < max_frames_to_track < total_frames:
        logger.info(
            f"Tracking only the first {max_frames_to_track} of {total_frames} frames"
        )
        total_frames = max_frames_to_track
    # Without YOLO metrics this returns plain fixed-length chunks
    # (a tail shorter than 10% of a chunk is merged into the last one)
    chunks = chunk_video_frames_adaptive(total_frames, fps, chunk_seconds)
    logger.info(
        f"{video_path.name}: {total_frames} frames at {fps:.2f} FPS -> "
        f"{len(chunks)} chunk(s) of {chunk_seconds:g}s, "
        f"grounding on the first {grounding_frames} frames of each"
        + (f", tracking 1 in {frame_stride} frames" if frame_stride > 1 else "")
    )
    n_processed = 0

    results_path = run_dir / f"{run_id}_tracking_outputs.parquet"
    chunk_dfs = []
    prev_tail = {}  # last frames of the previous chunk, for fallback seeding

    # Downscale large footage (e.g. 4K) as it is decoded, to avoid OOM
    transform = None
    height, width = meta.height, meta.width
    size = get_downscaled_size(height, width, max_short_side)
    if size is not None:
        transform = load_transforms(size)
        logger.info(
            f"Resize ON: source {width}x{height} -> target {size[1]}x{size[0]}; "
            "outputs are at the target resolution"
        )
    else:
        logger.info(
            f"Resize OFF: source {width}x{height} already "
            f"has short side <= {max_short_side}"
        )

    for chunk_idx, (start, end, _) in enumerate(chunks):
        logger.info(f"Chunk {chunk_idx}/{len(chunks) - 1}: frames {start}-{end}")
        t_chunk = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        frames = load_video_frames(
            video_path,
            start,
            end,
            loader="torchcodec",
            device=DECODE_DEVICE,
            transform=transform,
        )[::frame_stride]
        # The models see only the kept frames, numbered 0, 1, 2, ...;
        # frame_nums maps those back to video frame numbers
        frame_nums = range(start, end, frame_stride)
        n_processed += len(frames)
        h, w = frames[0].shape[:2]
        logger.info(f"  loaded {len(frames)} frames at {w}x{h}")

        # 1. Ground: text-prompted SAM 3 on the first frames, seed from the
        # frame with the most objects
        grounding_outputs = run_grounding(
            frames,
            0,
            math.ceil(grounding_frames / frame_stride),
            ground_chunk,
            cfg,
            device,
        )
        seed, masks, _, object_ids = find_best_grounding_frame(
            grounding_outputs, min_objects=min_objects, method="most_objects"
        )
        prefix = {}
        if seed is not None:
            # Frames before the seed frame keep the grounding outputs. The
            # tracker reuses the grounding IDs, so no remapping is needed.
            prefix = {
                i: grounding_outputs[i]
                for i in range(seed)
                if i in grounding_outputs
            }
            logger.info(
                f"  seeded from grounding frame {frame_nums[seed]} "
                f"({len(object_ids)} objects)"
            )
        else:
            # 2. Fallback: seed from the previous chunk's last good frame
            src_idx, masks, _, object_ids = find_frame_with_enough_objects(
                prev_tail, min_objects=min_objects, max_lookback=MAX_LOOKBACK_FRAMES
            )
            if src_idx is not None:
                seed = 0
                logger.warning(
                    f"  grounding found no frame with >= {min_objects} objects; "
                    f"seeded from previous-chunk frame {src_idx}"
                )
        del grounding_outputs

        # 3. Track from the seed frame to the end of the chunk
        points, seed_masks = _seed_prompts(masks, object_ids) if masks else ({}, {})
        prompt_masks = seed_masks if prompt == "masks" else None
        if points and tracker == "sam3.1":
            # Covers the whole chunk, so its outputs replace the prefix
            tracked = sam31.track(
                frames,
                0,
                points,
                seed_offset=seed,
                prompt_masks=prompt_masks,
            )
        elif points and tracker == "sam2-hq":
            tracked = track_chunk_sam2_hq(
                frames[seed:], seed, points, device, prompt_masks=prompt_masks
            )
        elif points:
            tracked = _process_tracker_chunk(
                frames[seed:],
                seed,
                points,
                cfg,
                device,
                tracker=tracker,
                prompt_masks=prompt_masks,
            )
        else:
            tracked = {}
            logger.warning(f"  no objects to track; chunk {chunk_idx} left empty")
        free_gpu_memory()

        for out in prefix.values():
            out["_model_type"] = GROUNDER_MODEL_TYPES[grounder]
        for out in tracked.values():
            out["_model_type"] = TRACKER_MODEL_TYPES[tracker]
        # Back to video frame numbers
        chunk_outputs = {
            frame_nums[i]: out for i, out in {**prefix, **tracked}.items()
        }
        for frame_idx, out in chunk_outputs.items():
            out["_chunk_idx"] = chunk_idx
            out["_is_chunk_start"] = frame_idx == start

        # 4. Save (rewritten every chunk, so a crash keeps finished chunks)
        if chunk_outputs:
            chunk_dfs.append(process_tracking_outputs(chunk_outputs).sort_index())
            pd.concat(chunk_dfs).to_parquet(results_path)

        prev_tail = {
            i: chunk_outputs[i]
            for i in sorted(chunk_outputs)[-MAX_LOOKBACK_FRAMES:]
        }
        peak_mem = (
            f"; peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
            if device.type == "cuda"
            else ""
        )
        logger.info(
            f"  chunk {chunk_idx} done in {time.time() - t_chunk:.0f}s{peak_mem}"
        )
        # Free this chunk before the next one is loaded
        del frames, chunk_outputs, tracked, prefix

    if sam31 is not None:
        sam31.close()
    elapsed = time.time() - t_start
    logger.info(
        f"Tracked {n_processed} of {total_frames} frames in {elapsed:.0f}s "
        f"({n_processed / elapsed:.1f} frames/s, including decoding)"
    )
    if not chunk_dfs:
        logger.warning("No objects tracked in any chunk; no parquet written")
        return
    logger.info(f"Tracking results saved to: {results_path}")

    # 5. Draw the tracks onto the source video
    if annotate:
        draw_annotations(
            video_path, results_path, run_dir / f"{run_id}_annotated_video.mp4"
        )


def main():
    parser = ArgumentParser(
        description="Barebones SAM 3 tracker: fixed chunks seeded by text grounding.",
        formatter_class=ArgumentDefaultsHelpFormatter,
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
        help="Video frames at the start of each chunk scanned for the seed frame.",
    )
    parser.add_argument(
        "--min-objects",
        type=int,
        default=10,
        help="Objects a grounding frame needs before it can seed the tracker.",
    )
    parser.add_argument(
        "--max-short-side",
        type=int,
        default=1080,
        help="Frames whose short side is larger are downscaled to it (target: 1080p).",
    )
    parser.add_argument(
        "--grounder",
        choices=list(GROUNDER_MODEL_TYPES),
        default="sam3",
        help="Text-prompted model that finds the objects in each chunk.",
    )
    parser.add_argument(
        "--tracker",
        choices=list(TRACKER_MODEL_TYPES),
        default="sam3",
        help="Promptable tracker that takes over from the grounding model.",
    )
    parser.add_argument(
        "--prompt",
        choices=["masks", "points"],
        default="masks",
        help="Hand each object to the tracker as its mask or as 3 click points.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the SAM 3.1 models: slow first run, faster tracking.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Track only every N-th frame (2: 25 -> 12.5 fps).",
    )
    parser.add_argument(
        "--no-annotated-video",
        action="store_true",
        help="Skip drawing the tracks onto the source video afterwards.",
    )
    parser.add_argument(
        "--max-frames-to-track",
        type=int,
        default=0,
        help="Track only the first N frames of the video; 0 tracks all of them.",
    )
    args = parser.parse_args()
    if args.max_frames_to_track < 0:
        parser.error("--max-frames-to-track must be >= 0")
    if args.frame_stride < 1:
        parser.error("--frame-stride must be >= 1")

    if not args.video_path.is_file():
        raise FileNotFoundError(f"Video not found: {args.video_path}")

    # Unique run id: all of this run's artefacts are named {run_id}_{artefact}
    run_id = get_run_id()
    run_dir = args.out_dir / sanitize_filename(args.video_path.stem)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logger(
        run_dir, job_type=f"{args.grounder}-{args.tracker}_simple", run_id=run_id
    )
    logger.info(f"Run id: {run_id}, logging to: {log_path}")
    logger.info(f"Arguments: {vars(args)}")

    track_video(
        video_path=args.video_path,
        run_dir=run_dir,
        run_id=run_id,
        text_prompt=args.text_prompt,
        chunk_seconds=args.chunk_seconds,
        grounding_frames=args.grounding_frames,
        min_objects=args.min_objects,
        max_short_side=args.max_short_side,
        max_frames_to_track=args.max_frames_to_track,
        tracker=args.tracker,
        prompt=args.prompt,
        grounder=args.grounder,
        compile=args.compile,
        frame_stride=args.frame_stride,
        annotate=not args.no_annotated_video,
    )


if __name__ == "__main__":
    main()
