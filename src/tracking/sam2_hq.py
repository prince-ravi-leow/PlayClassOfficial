"""HQ-SAM 2 (SysCV/sam-hq) point-prompted tracking for one chunk.

A drop-in for ``sam3._process_tracker_chunk``: same inputs (chunk frames,
global start index, click points per object) and the same per-frame output
dicts.

HQ-SAM 2 is not in transformers, so its code is imported from
``ext/sam-hq/sam-hq2`` (fetch it with ``pixi run -e tracker setup_hqsam2``).
The checkpoint (SAM 2.1 Large only) is downloaded from the Hugging Face Hub on
first use. SAM 2's optional CUDA extension is not built, so its small-hole
filling of masks is skipped (it warns once).
"""

import sys

from pathlib import Path
from unittest import mock

import numpy as np
import torch

from huggingface_hub import hf_hub_download
from loguru import logger

from .utils import FrameTensors, masks_to_frame_output

HQ_SAM2_DIR = Path(__file__).resolve().parents[2] / "ext" / "sam-hq" / "sam-hq2"
HQ_SAM2_REPO = "lkeab/hq-sam"
HQ_SAM2_CHECKPOINT = "sam2.1_hq_hiera_large.pt"
HQ_SAM2_CONFIG = "configs/sam2.1/sam2.1_hq_hiera_l.yaml"


def _build_predictor(device):
    """Build the HQ-SAM 2 video predictor from the ext/ checkout."""
    if not (HQ_SAM2_DIR / "sam2").is_dir():
        raise FileNotFoundError(
            f"HQ-SAM 2 code not found in {HQ_SAM2_DIR}. "
            "Run: pixi run -e tracker setup_hqsam2"
        )
    # HQ-SAM 2 ships as a fork of the `sam2` package; import it from the checkout
    if str(HQ_SAM2_DIR) not in sys.path:
        sys.path.insert(0, str(HQ_SAM2_DIR))
    from sam2.build_sam import build_sam2_hq_video_predictor

    checkpoint = hf_hub_download(HQ_SAM2_REPO, HQ_SAM2_CHECKPOINT)
    logger.info(f"Building HQ-SAM 2 video predictor from {checkpoint}")
    return build_sam2_hq_video_predictor(HQ_SAM2_CONFIG, checkpoint, device=device)


def _object_scores(inference_state: dict, frame_idx: int) -> np.ndarray:
    """Per-object presence probabilities SAM 2 stored for this frame."""
    output_dict = inference_state["output_dict"]
    for key in ("cond_frame_outputs", "non_cond_frame_outputs"):
        if frame_idx in output_dict[key]:
            logits = output_dict[key][frame_idx]["object_score_logits"]
            return torch.sigmoid(logits).flatten().float().cpu().numpy()
    return np.ones(len(inference_state["obj_ids"]), dtype=np.float32)


def track_chunk_sam2_hq(
    chunk_frames: list[np.ndarray],
    start_idx: int,
    all_prompt_points: dict[int, list],
    device,
    prompt_masks: dict[int, np.ndarray] | None = None,
) -> dict:
    """Track the objects clicked on ``chunk_frames[0]`` through the chunk.

    ``all_prompt_points`` maps object ID -> [[x, y], ...] positive clicks.
    ``prompt_masks`` (object ID -> (H, W) mask, same objects) prompts each
    object with its mask instead. Returns {global frame index: output dict},
    like ``_process_tracker_chunk``, with RLE masks.
    """
    device = torch.device(device)
    predictor = _build_predictor(device)
    height, width = chunk_frames[0].shape[:2]
    frames = FrameTensors(chunk_frames, predictor.image_size, device)

    outputs_per_frame = {}
    with (
        torch.inference_mode(),
        torch.autocast(device.type, dtype=torch.bfloat16),
    ):
        # init_state only reads videos from disk; hand it the in-memory frames
        with mock.patch(
            "sam2.sam2_hq_video_predictor.load_video_frames",
            return_value=(frames, height, width),
        ):
            inference_state = predictor.init_state(video_path="<in-memory>")

        kind = "mask" if prompt_masks is not None else "point"
        logger.info(
            f"Adding {kind} prompts for {len(all_prompt_points)} objects on frame 0"
        )
        for obj_id, points in all_prompt_points.items():
            if prompt_masks is not None:
                predictor.add_new_mask(
                    inference_state,
                    frame_idx=0,
                    obj_id=int(obj_id),
                    mask=np.asarray(prompt_masks[obj_id], dtype=bool),
                )
            else:
                predictor.add_new_points_or_box(
                    inference_state,
                    frame_idx=0,
                    obj_id=int(obj_id),
                    points=np.asarray(points, dtype=np.float32),
                    labels=np.ones(len(points), dtype=np.int32),
                )

        for local_idx, obj_ids, mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            masks = (mask_logits > 0.0).squeeze(1).cpu().numpy()  # (N, H, W)
            scores = _object_scores(inference_state, local_idx)
            outputs_per_frame[start_idx + local_idx] = masks_to_frame_output(
                masks, obj_ids, scores, rle=True
            )

    predictor.reset_state(inference_state)
    del inference_state, predictor
    return outputs_per_frame
