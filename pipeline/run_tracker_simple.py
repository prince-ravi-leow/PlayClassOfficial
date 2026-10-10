"""Barebones SAM 3 tracker: fixed-length chunks, each seeded by text grounding.

Settings come from ``config/tracker_simple.yaml`` (``--config`` for another
file); any flag below overrides its value.

For every fixed chunk (``--chunk-seconds``):

1. Run a text-prompted model (``--grounder``) on the first
   ``--grounding-frames`` frames and pick the one with the most objects
   (ties: least overlap, then highest score; ``find_best_grounding_frame``):

   - ``sam3``: ``Sam3VideoModel`` via transformers
   - ``sam3.1``: the SAM 3.1 video predictor (``ext/sam3`` submodule);
     ``--compile`` compiles the SAM 3.1 models (slow first run)
2. Hand each object to a promptable tracker (``--tracker``), as its mask
   (``--prompt masks``) or as 3 click points sampled inside it
   (``--prompt points``), and track from that frame to the end of the chunk:

   - ``sam3``: ``Sam3TrackerVideoModel``
   - ``sam2``: SAM 2.1 Large via transformers (``Sam2VideoModel``)
   - ``sam2-hq``: HQ-SAM 2 Large (SysCV/sam-hq); fetch its code first with
     ``pixi run -e tracker setup_hqsam2``
   - ``sam3.1``: the SAM 3.1 multiplex tracker (``ext/sam3`` submodule); it
     also tracks backwards from the chosen frame to the chunk start
3. Frames before the chosen frame keep the grounding model's own outputs
   (except with ``--tracker sam3.1``, which covers them).

If no grounding frame has ``--min-objects`` objects, the chunk is seeded from
the last frames of the previous chunk instead, or left empty if there is none.

With ``--parallel-chunks N`` > 1, N chunks are tracked at a time, each in its
own worker process with its own models (so N times the GPU memory). Chunks
then run independently: there is no fallback to the previous chunk. All of
them run on the GPU chosen with ``--gpu`` (0 or 1, as listed by nvidia-smi).

Once all chunks are tracked, object IDs are linked across chunks
(``link_chunk_ids`` in ``src/tracking/sam3_1.py``): each chunk's objects on
its first frame take the IDs of the best-overlapping of the previous chunk's
last ``--id-match-lookback-frames`` frames; objects with mask IoU below
``--id-match-min-iou`` get new IDs (logged). The parquet
saved after each chunk, in case of a crash, still has each chunk's own IDs.

With ``--frame-stride N`` only every N-th frame is tracked (e.g. 2: 25 -> 12.5
fps): the video frames whose numbers are multiples of N. The parquet only has
those frames; frame numbers stay those of the original video.

Frames whose short side is over ``--max-short-side`` pixels (e.g. 1080) are
downscaled while decoding (e.g. 4K -> 1080p), so all outputs, including mask
and box coordinates, are at the downscaled resolution.

Outputs, in ``{out_dir}/{video_stem}/``, prefixed with a unique run id
(timestamp, e.g. ``2026-10-06_18-28-51.37``):

- {run_id}_tracking_outputs.parquet (same schema as ``pipeline/run_tracker.py``)
- {run_id}_annotated_video.mp4 (masks, boxes and IDs drawn on the source video
  by ``pipeline/draw_annotations_simple.py``; skip with ``--no-annotated-video``)
- {run_id}_{grounder}-{tracker}_simple.log

Usage::

    pixi run -e tracker track_simple --video-path path/to/video.mp4 --tracker sam2
    pixi run -e tracker track_simple --video-path path/to/video.mp4 --gpu 1

SAM 3.1 needs the submodule (``git submodule update --init ext/sam3``) and
access to the gated ``facebook/sam3.1`` Hugging Face repo.
"""

import math
import multiprocessing
import os
import time

os.environ.setdefault(
    "PYTORCH_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.6"
)

from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    BooleanOptionalAction,
    Namespace,
)
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from loguru import logger
from omegaconf import OmegaConf

from pipeline.draw_annotations_simple import draw_annotations
from src._config import DEFAULT_CONFIG_DIR
from src.tracking.chunking import chunk_video_frames_adaptive
from src.tracking.grounding import find_best_grounding_frame, run_grounding
from src.tracking.sam2_hq import track_chunk_sam2_hq
from src.tracking.sam3 import _process_tracker_chunk, _process_video_chunk
from src.tracking.sam3_1 import Sam31, link_chunk_ids
from src.tracking.utils import (
    decode_frame_masks,
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
from src.utils.progress import ProgressDisplay, WorkerProgress, silence_tqdm

# Default settings; any of them can be overridden with a command-line flag
DEFAULT_CONFIG = Path(DEFAULT_CONFIG_DIR) / "tracker_simple.yaml"
# Share of the GPU's memory the parallel workers may use in total, split evenly.
# A worker over its share fails its chunk instead of squeezing out the others.
GPU_MEMORY_BUDGET = 0.9
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
# --prompt choices: hand each object to the tracker as its mask or click points
PROMPTS = ("masks", "points")


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


@dataclass
class ChunkSettings:
    """Everything needed to ground and track a chunk (picklable, for workers)."""

    video_path: Path
    size: tuple[int, int] | None  # (H, W) to downscale frames to, or None
    text_prompt: str
    grounding_frames: int
    min_objects: int
    grounder: str
    tracker: str
    prompt: str
    frame_stride: int
    compile: bool
    max_objects: int
    new_det_thresh: float
    grounding_batch_frames: int
    id_match_lookback_frames: int
    n_chunks: int


class ChunkTracker:
    """Grounds and tracks one chunk at a time, keeping its models between chunks."""

    def __init__(self, settings: ChunkSettings, make_bar=None):
        """``make_bar(desc, total)`` makes the progress bar of a pass over frames."""
        s = self.settings = settings
        self.make_bar = make_bar
        self.chunk_idx = 0
        # The _process_*_chunk helpers read their settings from an OmegaConf
        # config. No "tracking" key, so they keep the facebook/sam3 defaults.
        self.cfg = OmegaConf.create({"text_prompt": s.text_prompt})
        self.device = Accelerator().device
        # SAM 3.1 models are built on first use and kept for all chunks
        self.sam31 = None
        if "sam3.1" in (s.grounder, s.tracker):
            self.sam31 = Sam31(
                self.device,
                max_objects=s.max_objects,
                new_det_thresh=s.new_det_thresh,
                grounding_batch_frames=s.grounding_batch_frames,
                compile=s.compile,
                progress=self._make_chunk_bar if make_bar else None,
            )
        self.ground_chunk = (
            self.sam31.ground if s.grounder == "sam3.1" else _process_video_chunk
        )
        # Downscale large footage (e.g. 4K) as it is decoded, to avoid OOM
        self.transform = load_transforms(s.size) if s.size is not None else None

    def _make_chunk_bar(self, desc: str, total: int):
        return self.make_bar(f"chunk {self.chunk_idx} {desc}", total)

    def track(
        self, chunk_idx: int, start: int, end: int, prev_tail: dict
    ) -> tuple[pd.DataFrame | None, dict, int]:
        """Ground and track video frames [start, end).

        ``prev_tail`` has the previous chunk's last frames, to seed from if
        grounding finds too few objects ({} for none). Returns the chunk's
        parquet rows (None if nothing was tracked), its own last frames and
        the number of frames processed.
        """
        s = self.settings
        self.chunk_idx = chunk_idx
        tag = f"  [chunk {chunk_idx}]"
        logger.info(f"Chunk {chunk_idx}/{s.n_chunks - 1}: frames {start}-{end}")
        t_chunk = time.time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        # Keep the video frames whose numbers are multiples of the stride, so
        # the tracked frames form one grid whatever the chunk boundaries
        first = -(-start // s.frame_stride) * s.frame_stride
        frames = load_video_frames(
            s.video_path,
            start,
            end,
            loader="torchcodec",
            device=DECODE_DEVICE,
            transform=self.transform,
        )[first - start :: s.frame_stride]
        # The models see only the kept frames, numbered 0, 1, 2, ...;
        # frame_nums maps those back to video frame numbers
        frame_nums = range(first, end, s.frame_stride)
        h, w = frames[0].shape[:2]
        logger.info(f"{tag} loaded {len(frames)} frames at {w}x{h}")

        # 1. Ground: text-prompted SAM 3 on the first frames, seed from the
        # frame with the most objects
        grounding_outputs = run_grounding(
            frames,
            0,
            math.ceil(s.grounding_frames / s.frame_stride),
            self.ground_chunk,
            self.cfg,
            self.device,
        )
        seed, masks, _, object_ids = find_best_grounding_frame(
            grounding_outputs, min_objects=s.min_objects, method="most_objects"
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
                f"{tag} seeded from grounding frame {frame_nums[seed]} "
                f"({len(object_ids)} objects)"
            )
        else:
            # 2. Fallback: seed from the previous chunk's last good frame
            src_idx, masks, _, object_ids = find_frame_with_enough_objects(
                prev_tail,
                min_objects=s.min_objects,
                max_lookback=s.id_match_lookback_frames,
            )
            if src_idx is not None:
                seed = 0
                logger.warning(
                    f"{tag} grounding found no frame with >= {s.min_objects} "
                    f"objects; seeded from previous-chunk frame {src_idx}"
                )
        del grounding_outputs

        # 3. Track from the seed frame to the end of the chunk
        points, seed_masks = _seed_prompts(masks, object_ids) if masks else ({}, {})
        prompt_masks = seed_masks if s.prompt == "masks" else None
        if points and s.tracker == "sam3.1":
            # Covers the whole chunk, so its outputs replace the prefix
            tracked = self.sam31.track(
                frames, 0, points, seed_offset=seed, prompt_masks=prompt_masks
            )
        elif points and s.tracker == "sam2-hq":
            tracked = track_chunk_sam2_hq(
                frames[seed:], seed, points, self.device, prompt_masks=prompt_masks
            )
        elif points:
            tracked = _process_tracker_chunk(
                frames[seed:],
                seed,
                points,
                self.cfg,
                self.device,
                tracker=s.tracker,
                prompt_masks=prompt_masks,
                rle_masks=True,
            )
        else:
            tracked = {}
            logger.warning(f"{tag} no objects to track; chunk left empty")
        free_gpu_memory(log_stats=True)

        for out in prefix.values():
            out["_model_type"] = GROUNDER_MODEL_TYPES[s.grounder]
        for out in tracked.values():
            out["_model_type"] = TRACKER_MODEL_TYPES[s.tracker]
        # Back to video frame numbers
        chunk_outputs = {
            frame_nums[i]: out for i, out in {**prefix, **tracked}.items()
        }
        for frame_idx, out in chunk_outputs.items():
            out["_chunk_idx"] = chunk_idx
            out["_is_chunk_start"] = frame_idx == first
        rows = None
        if chunk_outputs:
            rows = process_tracking_outputs(chunk_outputs).sort_index()

        # Tracked masks are kept as RLE; fallback seeding needs arrays
        tail = {
            i: decode_frame_masks(chunk_outputs[i])
            for i in sorted(chunk_outputs)[-s.id_match_lookback_frames :]
        }
        peak_mem = (
            f"; peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
            if self.device.type == "cuda"
            else ""
        )
        logger.info(f"{tag} done in {time.time() - t_chunk:.0f}s{peak_mem}")
        return rows, tail, len(frames)

    def close(self) -> None:
        if self.sam31 is not None:
            self.sam31.close()


# Each worker process's ChunkTracker (see _init_worker)
_worker_tracker: ChunkTracker | None = None


def _init_worker(
    settings: ChunkSettings,
    log_dir: Path,
    job_type: str,
    run_id: str,
    events,
    memory_fraction: float,
) -> None:
    """Set up a worker process: cap its GPU memory, log to the run's log file,
    and send its progress bars and console log lines to the main process."""
    global _worker_tracker
    progress = WorkerProgress(events)
    silence_tqdm()  # libraries' own bars; the main process draws ours
    setup_logger(log_dir, job_type=job_type, run_id=run_id, console_sink=progress.log)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
    _worker_tracker = ChunkTracker(settings, make_bar=progress.bar)


def _track_chunk_in_worker(
    chunk_idx: int, start: int, end: int
) -> tuple[int, pd.DataFrame | None, int]:
    """Track one chunk in a worker; chunks run at the same time, so no fallback."""
    rows, _, n_processed = _worker_tracker.track(chunk_idx, start, end, prev_tail={})
    return chunk_idx, rows, n_processed


def track_video(
    video_path: Path,
    run_dir: Path,
    run_id: str,
    job_type: str,
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
    parallel_chunks: int = 1,
    max_objects: int = 64,
    new_det_thresh: float = 0.5,
    grounding_batch_frames: int = 4,
    id_match_lookback_frames: int = 5,
    id_match_min_iou: float = 0.5,
) -> None:
    """Track one video chunk by chunk, writing the parquet and annotated video.

    ``max_frames_to_track`` > 0 stops after that many frames; 0 tracks them all.
    ``grounder`` and ``tracker`` are keys of ``GROUNDER_MODEL_TYPES`` and
    ``TRACKER_MODEL_TYPES``; ``prompt`` is "masks" or "points". ``compile``
    applies ``torch.compile`` to the SAM 3.1 models. ``frame_stride`` > 1
    tracks only every ``frame_stride``-th frame. ``annotate`` draws the tracks
    onto the source video afterwards. ``parallel_chunks`` > 1 tracks that many
    chunks at a time, each worker process with its own models (logging to the
    run's log as ``{run_dir}/{run_id}_{job_type}.log``); chunks then cannot
    fall back on the previous chunk's last frames. ``max_objects``,
    ``new_det_thresh`` and ``grounding_batch_frames`` set up SAM 3.1
    grounding (see ``Sam31``); ``id_match_lookback_frames`` and
    ``id_match_min_iou`` the cross-chunk ID linking (see ``link_chunk_ids``).
    """
    t_start = time.time()
    # One torchcodec decoder supplies the metadata (workers open their own)
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

    size = get_downscaled_size(meta.height, meta.width, max_short_side)
    if size is not None:
        logger.info(
            f"Resize ON: source {meta.width}x{meta.height} -> target "
            f"{size[1]}x{size[0]}; outputs are at the target resolution"
        )
    else:
        logger.info(
            f"Resize OFF: source {meta.width}x{meta.height} already "
            f"has short side <= {max_short_side}"
        )
    settings = ChunkSettings(
        video_path=video_path,
        size=size,
        text_prompt=text_prompt,
        grounding_frames=grounding_frames,
        min_objects=min_objects,
        grounder=grounder,
        tracker=tracker,
        prompt=prompt,
        frame_stride=frame_stride,
        compile=compile,
        max_objects=max_objects,
        new_det_thresh=new_det_thresh,
        grounding_batch_frames=grounding_batch_frames,
        id_match_lookback_frames=id_match_lookback_frames,
        n_chunks=len(chunks),
    )

    results_path = run_dir / f"{run_id}_tracking_outputs.parquet"
    rows_by_chunk = {}
    n_processed = 0

    def save(chunk_idx: int, rows: pd.DataFrame | None) -> None:
        # Rewritten after every chunk, so a crash keeps the finished chunks
        if rows is not None:
            rows_by_chunk[chunk_idx] = rows
            pd.concat([rows_by_chunk[i] for i in sorted(rows_by_chunk)]).to_parquet(
                results_path
            )

    if parallel_chunks <= 1:
        chunk_tracker = ChunkTracker(settings)
        prev_tail = {}  # last frames of the previous chunk, for fallback seeding
        for chunk_idx, (start, end, _) in enumerate(chunks):
            rows, prev_tail, n = chunk_tracker.track(chunk_idx, start, end, prev_tail)
            n_processed += n
            save(chunk_idx, rows)
        chunk_tracker.close()
    else:
        logger.info(
            f"Tracking {parallel_chunks} chunks at a time, each in its own process "
            "(no fallback seeding from the previous chunk)"
        )
        # Spawned, not forked: CUDA cannot be shared with forked processes
        context = multiprocessing.get_context("spawn")
        events = context.Queue()  # the workers' progress bars and log lines
        with (
            ProgressDisplay(events),
            ProcessPoolExecutor(
                parallel_chunks,
                mp_context=context,
                initializer=_init_worker,
                initargs=(
                    settings,
                    run_dir,
                    job_type,
                    run_id,
                    events,
                    GPU_MEMORY_BUDGET / parallel_chunks,
                ),
            ) as pool,
        ):
            futures = [
                pool.submit(_track_chunk_in_worker, chunk_idx, start, end)
                for chunk_idx, (start, end, _) in enumerate(chunks)
            ]
            for future in as_completed(futures):
                chunk_idx, rows, n = future.result()
                n_processed += n
                save(chunk_idx, rows)

    elapsed = time.time() - t_start
    logger.info(
        f"Tracked {n_processed} of {total_frames} frames in {elapsed:.0f}s "
        f"({n_processed / elapsed:.1f} frames/s, including decoding)"
    )
    if not rows_by_chunk:
        logger.warning("No objects tracked in any chunk; no parquet written")
        return
    # 4. Same object, same ID in every chunk
    linked = link_chunk_ids(
        rows_by_chunk,
        lookback_frames=id_match_lookback_frames,
        min_iou=id_match_min_iou,
    )
    pd.concat([linked[i] for i in sorted(linked)]).to_parquet(results_path)
    logger.info(f"Tracking results saved to: {results_path}")

    # 5. Draw the tracks onto the source video
    if annotate:
        draw_annotations(
            video_path, results_path, run_dir / f"{run_id}_annotated_video.mp4"
        )


def parse_args() -> Namespace:
    """Settings from the ``--config`` YAML; flags on the command line override it."""
    config_parser = ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML file with the settings; the flags below override its values.",
    )
    config_path = config_parser.parse_known_args()[0].config
    cfg = OmegaConf.to_container(OmegaConf.load(config_path))

    # No defaults here: they all come from the config (shown by --help)
    parser = ArgumentParser(
        description="Barebones SAM 3 tracker: fixed chunks seeded by text grounding.",
        formatter_class=ArgumentDefaultsHelpFormatter,
        parents=[config_parser],
    )
    parser.add_argument("--video-path", type=Path, help="Input video file.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output root; results go in {out_dir}/{video_stem}/.",
    )
    parser.add_argument("--text-prompt", help="What SAM 3 should segment.")
    parser.add_argument(
        "--chunk-seconds", type=float, help="Chunk length in seconds."
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="CUDA GPU to run on, as numbered by nvidia-smi (e.g. 0 or 1).",
    )
    parser.add_argument(
        "--parallel-chunks",
        type=int,
        help="Chunks tracked at a time, each in its own process with its own models.",
    )
    parser.add_argument(
        "--grounding-frames",
        type=int,
        help="Video frames at the start of each chunk scanned for the seed frame.",
    )
    parser.add_argument(
        "--min-objects",
        type=int,
        help="Objects a grounding frame needs before it can seed the tracker.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        help="SAM 3.1 grounding: most objects it tracks (upstream: 16).",
    )
    parser.add_argument(
        "--new-det-thresh",
        type=float,
        help="SAM 3.1 grounding: detection score a new object needs (upstream: 0.65).",
    )
    parser.add_argument(
        "--grounding-batch-frames",
        type=int,
        help="SAM 3.1 grounding: frames detected at once; fewer use less GPU memory.",
    )
    parser.add_argument(
        "--id-match-lookback-frames",
        type=int,
        help="Previous chunk's last frames matched against each chunk's first frame.",
    )
    parser.add_argument(
        "--id-match-min-iou",
        type=float,
        help="Mask IoU an object needs with its match to keep its ID across chunks.",
    )
    parser.add_argument(
        "--max-short-side",
        type=int,
        help="Frames whose short side is larger are downscaled to it (target: 1080p).",
    )
    parser.add_argument(
        "--grounder",
        choices=list(GROUNDER_MODEL_TYPES),
        help="Text-prompted model that finds the objects in each chunk.",
    )
    parser.add_argument(
        "--tracker",
        choices=list(TRACKER_MODEL_TYPES),
        help="Promptable tracker that takes over from the grounding model.",
    )
    parser.add_argument(
        "--prompt",
        choices=PROMPTS,
        help="Hand each object to the tracker as its mask or as 3 click points.",
    )
    parser.add_argument(
        "--compile",
        action=BooleanOptionalAction,
        help="torch.compile the SAM 3.1 models: slow first run, faster tracking.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        help="Track only every N-th frame (2: 25 -> 12.5 fps).",
    )
    parser.add_argument(
        "--annotated-video",
        action=BooleanOptionalAction,
        help="Draw the tracks onto the source video afterwards.",
    )
    parser.add_argument(
        "--max-frames-to-track",
        type=int,
        help="Track only the first N frames of the video; 0 tracks all of them.",
    )

    # Config keys must match the flags one to one (dashes -> underscores)
    settings = set(vars(parser.parse_args([]))) - {"config"}
    if unknown := set(cfg) - settings:
        parser.error(f"unknown key(s) in {config_path}: {sorted(unknown)}")
    if missing := settings - set(cfg):
        parser.error(f"key(s) missing from {config_path}: {sorted(missing)}")
    parser.set_defaults(**cfg)
    args = parser.parse_args()

    # argparse checks choices only for flags, not for defaults from the config
    for name, choices in [
        ("grounder", GROUNDER_MODEL_TYPES),
        ("tracker", TRACKER_MODEL_TYPES),
        ("prompt", PROMPTS),
    ]:
        if getattr(args, name) not in choices:
            parser.error(f"{name} must be one of {list(choices)}")
    if args.video_path is None:
        parser.error(f"set video_path in {config_path} or pass --video-path")
    if args.max_frames_to_track < 0:
        parser.error("--max-frames-to-track must be >= 0")
    if args.frame_stride < 1:
        parser.error("--frame-stride must be >= 1")
    if args.gpu < 0:
        parser.error("--gpu must be >= 0")
    for name in (
        "parallel_chunks",
        "max_objects",
        "grounding_batch_frames",
        "id_match_lookback_frames",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be >= 1")
    for name in ("new_det_thresh", "id_match_min_iou"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"{name} must be between 0 and 1")
    return args


def main():
    args = parse_args()
    # Pick the GPU before anything touches CUDA; the workers inherit it (and
    # see it as cuda:0)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if torch.cuda.device_count() == 0:
        raise SystemExit(f"No CUDA GPU {args.gpu} found (see nvidia-smi)")
    if not args.video_path.is_file():
        raise FileNotFoundError(f"Video not found: {args.video_path}")

    # Unique run id: all of this run's artefacts are named {run_id}_{artefact}
    run_id = get_run_id()
    run_dir = args.out_dir / sanitize_filename(args.video_path.stem)
    run_dir.mkdir(parents=True, exist_ok=True)
    job_type = f"{args.grounder}-{args.tracker}_simple"
    log_path = setup_logger(run_dir, job_type=job_type, run_id=run_id)
    logger.info(f"Run id: {run_id}, logging to: {log_path}")
    logger.info(f"Arguments: {vars(args)}")

    track_video(
        video_path=args.video_path,
        run_dir=run_dir,
        run_id=run_id,
        job_type=job_type,
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
        annotate=args.annotated_video,
        parallel_chunks=args.parallel_chunks,
        max_objects=args.max_objects,
        new_det_thresh=args.new_det_thresh,
        grounding_batch_frames=args.grounding_batch_frames,
        id_match_lookback_frames=args.id_match_lookback_frames,
        id_match_min_iou=args.id_match_min_iou,
    )


if __name__ == "__main__":
    main()
