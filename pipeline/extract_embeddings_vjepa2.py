"""
Extract video-model embeddings (V-JEPA 2 / 2.1) from tracked objects.

Unlike DINOv3 extraction (per-frame CLS tokens), this feeds K frames per
window as a single video clip through a spatiotemporal backbone and
mean-pools the output tokens into one embedding per window.

V-JEPA 2.1 setup (one-time)::

    bash scripts/setup_vjepa2.1.sh

This downloads the ViT-B and ViT-L checkpoints and patches the torch.hub cache to rename
``src/`` to ``vjepa2/`` (avoids collision with this project's ``src/``).
"""

import sys

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch

from loguru import logger
from transformers import AutoModel, AutoVideoProcessor

from src._config import (
    DEFAULT_DATASET_DIR,
    DEFAULT_TRACKING_DIR,
    DEFAULT_VIDEO_DIR,
)
from src.dataset.crops import CROP_MODES
from src.dataset.embeddings.vjepa2 import (
    VJEPA21Wrapper,
    extract_video_embeddings,
    is_hub_model,
)
from src.dataset.utils import assert_embedding_label_alignment, resolve_video_path
from src.utils.memory import free_gpu_memory


def parse_args():
    parser = ArgumentParser(
        description="Extract video-model embeddings from tracked objects."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
    )
    parser.add_argument(
        "--tracking-model",
        type=str,
        default="sam3_best",
        help="Tracking model subdirectory under the tracking results dir.",
    )
    parser.add_argument(
        "--tracking-dir",
        type=Path,
        default=None,
        help="Override: explicit tracking dir (ignores --tracking-model).",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        nargs="+",
        default=None,
        help=f"Directory(ies) with .mp4 files (default: all subdirs of {DEFAULT_VIDEO_DIR}/)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/vjepa2-vitl-fpc64-256",
        help="HF model name, or torch.hub entry for V-JEPA 2.1 "
        "(e.g. 'vjepa2_1_vit_large_384')",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device (e.g. cuda:0, cuda:1, cpu)"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=64,
        help="Frames per clip (must match model's pretrained frame count, default: 64)",
    )
    parser.add_argument(
        "--crop-mode",
        type=str,
        choices=list(CROP_MODES),
        default="bbox",
        help="Crop strategy: 'bbox'=per-frame bbox crop (default), "
        "'plain256'=fixed 256x256 around bbox centroid, "
        "'plain384'=fixed 384x384 around bbox centroid, "
        "'union512'=fixed 512x512 around union centroid, "
        "'darken512'=union512 with darkened background (untested), "
        "'roi512'=union512 + ROI patch indices (untested)",
    )
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="Output per-timestep spatial-mean-pooled tokens (T, D) instead of a single (D,) vector",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Save raw patch tokens (T*S, D) per window instead of pooled",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output filename (default: auto-generated from model name)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing output file and skip videos already extracted",
    )
    parser.add_argument(
        "--cid-checkpoint",
        type=Path,
        default=None,
        help="Path to CID checkpoint (encoder_only.pt) to load adapted weights",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only 1 video and 2 windows, skip save. For smoke-testing.",
    )
    return parser.parse_args()


def _build_output_name(args):
    """Auto-generate output filename from model config.

    Examples:
        vjepa2_1_vit_large_384 -> embeddings_vjepa21_vitl
        vjepa2_1_vit_base_384  -> embeddings_vjepa21_vitb
        facebook/vjepa2-vitl-fpc64-256 -> embeddings_vjepa2_vitl
    """
    from src.dataset.utils import parse_model_size

    m = args.model_name.lower()

    # Model family
    family = (
        "vjepa21"
        if "vjepa2_1" in m or "vjepa2.1" in m
        else "vjepa2" if "vjepa2" in m else "video"
    )

    size = parse_model_size(args.model_name)
    if size is None:
        raise ValueError(f"Cannot infer model size from name: {args.model_name}")

    parts = ["embeddings", family, size]
    if hasattr(args, "cid_checkpoint") and args.cid_checkpoint:
        parts.append("cid")
    if args.num_frames != 64:
        parts.append(f"f{args.num_frames}")
    if args.crop_mode != "bbox":
        parts.append(args.crop_mode)
    if args.raw:
        parts.append("raw")
    elif args.temporal:
        parts.append("temporal")
    return "_".join(parts) + ".pt"


def main():
    args = parse_args()

    if args.tracking_dir is None:
        args.tracking_dir = Path(DEFAULT_TRACKING_DIR) / args.tracking_model

    tracks_path = args.dataset_dir / "tracks.parquet"
    if not tracks_path.exists():
        logger.error(f"tracks.parquet not found in {args.dataset_dir}")
        sys.exit(1)

    # Resolve video dirs
    if args.video_dir is None:
        root = Path(DEFAULT_VIDEO_DIR)
        args.video_dir = sorted(p for p in root.iterdir() if p.is_dir())
        logger.info(f"Auto-discovered {len(args.video_dir)} video dir(s) under {root}")

    load_cols = ["video_id", "bird_id", "frame_idx", "window", "bbox"]
    tracks = pd.read_parquet(tracks_path, columns=load_cols)
    video_ids = sorted(tracks["video_id"].unique())
    logger.info(f"Loaded {len(tracks)} track rows across {len(video_ids)} video(s)")

    if args.dry_run:
        video_ids = video_ids[:1]
        first_windows = tracks[tracks["video_id"] == video_ids[0]]["window"].unique()[
            :2
        ]
        tracks = tracks[
            (tracks["video_id"] == video_ids[0])
            & (tracks["window"].isin(first_windows))
        ]
        logger.info(
            f"Dry run: 1 video, {len(first_windows)} window(s), {len(tracks)} rows"
        )

    # Load model + processor
    device = torch.device(args.device)
    logger.info(f"Loading model: {args.model_name}")

    if is_hub_model(args.model_name):
        # V-JEPA 2.1 via torch.hub (no HF checkpoint yet).
        processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
        encoder, _predictor = torch.hub.load("facebookresearch/vjepa2", args.model_name)
        del _predictor
        if args.cid_checkpoint:
            state_dict = torch.load(
                args.cid_checkpoint, map_location="cpu", weights_only=True
            )
            encoder.load_state_dict(state_dict["ema_encoder"], strict=True)
            logger.info(f"Loaded CID checkpoint from {args.cid_checkpoint}")
        model = VJEPA21Wrapper(encoder, device)
        d_model = encoder.embed_dim
    else:
        processor = AutoVideoProcessor.from_pretrained(args.model_name)
        model = AutoModel.from_pretrained(args.model_name, dtype=torch.bfloat16)
        model = model.to(device).eval()
        d_model = model.config.hidden_size

    logger.info(f"Model on {device}, hidden_size={d_model}")

    # Extract per video
    # For raw mode, save per-video to avoid OOM (8192 x 1024 x ~500 windows = ~4 GB/video)
    output_name = args.output_name or _build_output_name(args)
    output_path = (
        Path(output_name) if "/" in str(output_name) else args.dataset_dir / output_name
    )
    all_embeddings = {}
    if args.resume and not args.dry_run and output_path.exists():
        all_embeddings = torch.load(output_path, map_location="cpu", weights_only=False)
        logger.info(f"Resuming from {output_path}: {len(all_embeddings)} embeddings")
    done_videos = {k[0] for k in all_embeddings}

    for video_id in video_ids:
        if video_id in done_videos:
            logger.info(f"--- Skipping {video_id} (already extracted) ---")
            continue
        logger.info(f"--- Processing video: {video_id} ---")
        video_path = resolve_video_path(video_id, args.tracking_dir, args.video_dir)
        if video_path is None:
            logger.error(f"Skipping {video_id}: video file not found")
            continue

        video_tracks = tracks[tracks["video_id"] == video_id]
        emb_dict = extract_video_embeddings(
            video_tracks,
            video_path,
            model,
            processor,
            device,
            num_frames=args.num_frames,
            crop_mode=args.crop_mode,
            temporal=args.temporal,
            raw=args.raw,
        )
        all_embeddings.update(emb_dict)
        logger.info(f"  {len(emb_dict)} window embeddings extracted")

        if not args.dry_run:
            # Checkpoint after every video: these runs are hours long, and a crash
            # near the end would otherwise discard everything.
            torch.save(all_embeddings, output_path)
            logger.info(f"  Saved {len(all_embeddings)} total embeddings (incremental)")

    if not all_embeddings:
        raise ValueError("No embeddings extracted.")

    if args.dry_run:
        sample_key = next(iter(all_embeddings))
        logger.info(
            f"Dry run complete: {len(all_embeddings)} groups, "
            f"sample shape {all_embeddings[sample_key].shape}"
        )
    else:
        # Check alignment with labels
        assert_embedding_label_alignment(set(all_embeddings.keys()), args.dataset_dir)

        # Save
        torch.save(all_embeddings, output_path)
        logger.info(f"Saved {len(all_embeddings)} embeddings to {output_path}")

    del model, processor
    free_gpu_memory()
    logger.info("Done.")


if __name__ == "__main__":
    main()
