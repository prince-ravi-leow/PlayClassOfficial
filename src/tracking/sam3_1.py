"""SAM 3.1 (facebookresearch/sam3, "Object Multiplex") grounding and tracking.

Used chunk by chunk by ``pipeline/run_tracker_simple.py``:

- ``Sam31.ground``: the text-prompted SAM 3.1 video predictor (detector and
  tracker) on the grounding frames. Same signature and output dicts as
  ``sam3._process_video_chunk``. Its object cap (``max_objects``), detection
  score threshold for new objects (``new_det_thresh``) and detector batch
  (``grounding_batch_frames``) come from ``config/tracker_simple.yaml``.
- ``Sam31.track``: the SAM 3.1 multiplex tracker on its own, seeded on one
  frame with click points or masks and run forwards and backwards from it.
  Output dicts match ``sam3._process_tracker_chunk``. All objects share one
  multiplex state, so they are tracked jointly. Clicks are first turned into
  masks one object at a time, as the SAM 3.1 predictor does: clicking several
  objects into one multiplex state gives much worse masks.

``link_chunk_ids`` then gives objects the same IDs across chunks: each
chunk's objects on its first frame are matched against the previous chunk's
last frames by mask IoU (see its docstring).

With ``compile=True`` both models are compiled with ``torch.compile``
(upstream: about 2x faster; the tracker without CUDA graphs, see
``_compile_tracker``). The first chunk is slow while it compiles.

The code is imported from the ``ext/sam3`` submodule
(``git submodule update --init ext/sam3``) and the weights come from the
gated ``facebook/sam3.1`` Hugging Face repo. Works around three upstream bugs
(sam3 commit 0570b3a):

1. The predictor's ``start_session`` passes ``offload_state_to_cpu``, which
   the multiplex model's ``init_state`` does not accept.
2. The standalone tracker asks its backbone for detector features, which
   ``forward_image`` then fails to unpack; the tracker does not use them.
3. Propagating with ``max_frame_num_to_track`` crashes text grounding, so the
   grounding session holds only the grounding frames.
"""

import sys

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import torch

from huggingface_hub import hf_hub_download
from loguru import logger
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from .utils import FrameTensors, masks_to_frame_output

SAM3_DIR = Path(__file__).resolve().parents[2] / "ext" / "sam3"
SAM31_REPO = "facebook/sam3.1"
SAM31_CHECKPOINT = "sam3.1_multiplex.pt"
# SAM 3.1 input normalisation (fixed by its training)
IMAGE_MEAN = IMAGE_STD = (0.5, 0.5, 0.5)


def _compile_tracker(tracker) -> None:
    """Compile the tracker's heavy parts with ``torch.compile``.

    The same parts upstream's ``compile=True`` compiles (memory encoder,
    memory attention, mask decoder, image backbone), but without CUDA graphs:
    those reuse their output buffers on every call, while the tracker keeps
    mask decoder outputs from frame to frame, so upstream's version fails with
    "accessing tensor output of CUDAGraphs that has been overwritten".
    """
    # Room for the shapes that vary with the number of memories (as upstream)
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 2048
    mode = "max-autotune-no-cudagraphs"
    trunk = tracker.backbone.vision_backbone.trunk
    trunk.forward = torch.compile(trunk.forward, mode=mode, fullgraph=True)
    for module, dynamic in (
        (tracker.maskmem_backbone, False),
        (tracker.transformer.encoder, True),  # number of memories varies
        (tracker.sam_mask_decoder, False),
    ):
        module.forward = torch.compile(
            module.forward, mode=mode, fullgraph=True, dynamic=dynamic
        )


def _model_builder():
    """The ``sam3.model_builder`` module, imported from the submodule."""
    if not (SAM3_DIR / "sam3").is_dir():
        raise FileNotFoundError(
            f"SAM 3 code not found in {SAM3_DIR}. "
            "Run: git submodule update --init ext/sam3"
        )
    if str(SAM3_DIR) not in sys.path:
        sys.path.insert(0, str(SAM3_DIR))
    import sam3.model_builder

    return sam3.model_builder


class Sam31:
    """SAM 3.1 grounding and tracking; each model is built on first use.

    Grounding settings: ``max_objects`` the predictor tracks (upstream: 16),
    the detection score a new object needs (``new_det_thresh``, upstream:
    0.65) and the frames the detector processes at once
    (``grounding_batch_frames``, upstream: 16; fewer need less GPU memory).
    ``progress(desc, total)`` makes a tqdm-like bar (``update``, ``close``)
    for the frames of each grounding and tracking pass; by default none shows.

    Building the predictor turns on bf16 autocast for the rest of the process
    (upstream behaviour); ``close`` turns it off again.
    """

    def __init__(
        self,
        device,
        *,
        max_objects: int,
        new_det_thresh: float,
        grounding_batch_frames: int,
        compile: bool = False,
        progress: Callable[[str, int], Any] | None = None,
    ):
        self.device = torch.device(device)
        self.progress = progress or (
            lambda desc, total: tqdm(total=total, desc=desc, disable=True)
        )
        self.max_objects = max_objects
        self.new_det_thresh = new_det_thresh
        self.grounding_batch_frames = grounding_batch_frames
        self.compile = compile
        self._predictor = None
        self._tracker = None

    @property
    def predictor(self):
        """Text-prompted video predictor (detector + tracker)."""
        if self._predictor is None:
            logger.info("Building SAM 3.1 video predictor")
            predictor = _model_builder().build_sam3_multiplex_video_predictor(
                use_fa3=False,  # FlashAttention 3 needs a Hopper GPU
                max_num_objects=self.max_objects,
                compile=self.compile,
            )
            predictor.model.new_det_thresh = self.new_det_thresh
            predictor.model.batched_grounding_batch_size = self.grounding_batch_frames
            # Upstream bug 1
            init_state = predictor.model.init_state
            predictor.model.init_state = lambda offload_state_to_cpu=False, **kwargs: (
                init_state(**kwargs)
            )
            self._predictor = predictor
        return self._predictor

    @property
    def tracker(self):
        """Multiplex tracker with its own image backbone, for prompted tracking."""
        if self._tracker is None:
            logger.info("Building SAM 3.1 multiplex tracker")
            tracker = _model_builder().build_sam3_multiplex_video_model(
                load_from_HF=False,
                use_fa3=False,
                # Real-valued RoPE, as upstream uses for compiling
                use_rope_real=self.compile,
                device=self.device,
            )
            # The checkpoint holds the whole predictor: keep the tracker and
            # the image backbone it shares with the detector
            checkpoint = torch.load(
                hf_hub_download(SAM31_REPO, SAM31_CHECKPOINT),
                map_location="cpu",
                weights_only=True,
            )
            prefixes = {
                "tracker.model.": "",
                "detector.backbone.vision_backbone.": "backbone.vision_backbone.",
            }
            state_dict = {
                new + key[len(old) :]: value
                for key, value in checkpoint.items()
                for old, new in prefixes.items()
                if key.startswith(old)
            }
            missing, unexpected = tracker.load_state_dict(state_dict, strict=False)
            # Only the real-valued RoPE tables (computed at build time) may be missing
            if unexpected or any("freqs_cis" not in key for key in missing):
                raise RuntimeError(
                    f"SAM 3.1 tracker weights do not match: missing {missing}, "
                    f"unexpected {unexpected}"
                )
            tracker.eval()
            if self.compile:
                _compile_tracker(tracker)
            # Upstream bug 2
            forward_image = tracker.forward_image
            tracker.forward_image = lambda img, need_sam3_out=False, **kwargs: (
                forward_image(img, need_sam3_out=False, **kwargs)
            )
            self._tracker = tracker
        return self._tracker

    def ground(self, chunk_frames, start_idx, cfg, _device=None) -> dict:
        """Text-prompted detection and tracking on ``chunk_frames``.

        Same signature as ``sam3._process_video_chunk`` (``cfg.text_prompt``
        is the prompt; the device argument is unused). Returns {global frame
        index: output dict}.
        """
        predictor = self.predictor
        session_id = predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": [Image.fromarray(f) for f in chunk_frames],
            }
        )["session_id"]
        predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": cfg.text_prompt,
            }
        )
        outputs_per_frame = {}
        bar = self.progress("grounding", len(chunk_frames))
        for response in predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
            }
        ):
            out = response["outputs"]
            outputs_per_frame[start_idx + response["frame_index"]] = (
                masks_to_frame_output(
                    out["out_binary_masks"], out["out_obj_ids"], out["out_probs"]
                )
            )
            bar.update()
        bar.close()
        predictor.handle_request({"type": "close_session", "session_id": session_id})
        return outputs_per_frame

    def _click_masks(self, frame: np.ndarray, all_prompt_points: dict) -> list:
        """Each object's mask from its clicks, segmented one object at a time."""
        height, width = frame.shape[:2]
        masks = []
        for obj_id, points in all_prompt_points.items():
            state = self.tracker.init_state(
                video_height=height, video_width=width, num_frames=1
            )
            state["images"] = FrameTensors(
                [frame], self.tracker.image_size, self.device, IMAGE_MEAN, IMAGE_STD
            )
            _, _, _, video_res_masks = self.tracker.add_new_points(
                state,
                0,
                int(obj_id),
                torch.tensor(np.asarray(points) / [width, height], dtype=torch.float32),
                torch.ones(len(points), dtype=torch.int32),
                clear_old_points=True,
            )
            masks.append(video_res_masks[0, 0] > 0)
        return masks

    def track(
        self,
        chunk_frames: list[np.ndarray],
        start_idx: int,
        all_prompt_points: dict[int, list],
        seed_offset: int = 0,
        prompt_masks: dict[int, np.ndarray] | None = None,
    ) -> dict:
        """Track objects through ``chunk_frames`` from the seed frame both ways.

        Objects are given on ``chunk_frames[seed_offset]`` as click points
        (``all_prompt_points``: object ID -> [[x, y], ...]) or, if given, as
        masks (``prompt_masks``: object ID -> (H, W) mask, same objects).
        Returns {global frame index: output dict} for every frame of the chunk,
        with RLE masks (see ``masks_to_frame_output``).
        """
        tracker = self.tracker
        height, width = chunk_frames[0].shape[:2]
        obj_ids = [int(oid) for oid in all_prompt_points]
        outputs_per_frame = {}
        with (
            torch.inference_mode(),
            torch.autocast(self.device.type, dtype=torch.bfloat16),
        ):
            if prompt_masks is None:
                masks = self._click_masks(chunk_frames[seed_offset], all_prompt_points)
            else:
                masks = [torch.as_tensor(prompt_masks[oid]) for oid in obj_ids]
            state = tracker.init_state(
                video_height=height, video_width=width, num_frames=len(chunk_frames)
            )
            state["images"] = FrameTensors(
                chunk_frames, tracker.image_size, self.device, IMAGE_MEAN, IMAGE_STD
            )
            kind = "mask" if prompt_masks is not None else "point"
            logger.info(
                f"Adding {kind} prompts for {len(obj_ids)} objects "
                f"on frame {seed_offset}"
            )
            tracker.add_new_masks(state, seed_offset, obj_ids, torch.stack(masks))
            tracker.propagate_in_video_preflight(state, run_mem_encoder=True)

            bar = self.progress("tracking", len(chunk_frames))
            for reverse in (False, True):
                if reverse and seed_offset == 0:
                    break
                for (
                    local_idx,
                    ids,
                    _,
                    video_res_masks,
                    score_logits,
                ) in tracker.propagate_in_video(
                    state,
                    start_frame_idx=seed_offset,
                    max_frame_num_to_track=len(chunk_frames),
                    reverse=reverse,
                ):
                    outputs_per_frame[start_idx + local_idx] = masks_to_frame_output(
                        (video_res_masks > 0).squeeze(1).cpu().numpy(),
                        ids,
                        torch.sigmoid(score_logits.float()).flatten().cpu().numpy(),
                        rle=True,
                    )
                    bar.update()
            bar.close()
        del state
        return outputs_per_frame

    def close(self) -> None:
        """Free the models and undo the predictor's global bf16 autocast."""
        if self._predictor is not None:
            self._predictor.bf16_context.__exit__(None, None, None)
            self._predictor.shutdown()
        self._predictor = self._tracker = None


def _frame_rles(rows: pd.DataFrame, frame_idx: int) -> tuple[list, list[int]]:
    """One frame's masks as COCO RLE dicts, and their object IDs."""
    frame = rows.xs(frame_idx, level="frame_idx")
    rles = [
        {"size": list(size), "counts": counts.encode()}
        for counts, size in zip(frame["counts"], frame["size"])
    ]
    return rles, frame.index.tolist()


def link_chunk_ids(
    rows_by_chunk: dict[int, pd.DataFrame], *, lookback_frames: int, min_iou: float
) -> dict[int, pd.DataFrame]:
    """Give each object the same ID in every chunk (parquet rows, by chunk index).

    Chunk by chunk, the objects on a chunk's first frame are paired one to one
    with the objects of each of the previous chunk's last ``lookback_frames``
    frames, maximising mask IoU (scipy's ``linear_sum_assignment``). The frame
    with the highest mean IoU per object (0 for an unpaired object) wins, and
    its IDs carry over. An object whose IoU with its partner is below
    ``min_iou`` (or that has none) gets an ID not used in any chunk so far;
    this is logged. The first chunk keeps its IDs; a chunk after an empty one
    gets all new IDs.
    """
    linked = {}
    next_id = 0  # lowest ID not used in any chunk so far
    for chunk_idx in sorted(rows_by_chunk):
        rows = rows_by_chunk[chunk_idx]
        tag = f"  [chunk {chunk_idx}]"
        object_ids = rows.index.get_level_values("object_id").unique().tolist()
        if not linked:
            id_map = {oid: oid for oid in object_ids}
        else:
            # Why each object without a carried-over ID gets a new one
            reasons = dict.fromkeys(object_ids, "no match")
            id_map = {}
            prev = linked.get(chunk_idx - 1)
            if prev is None:
                reasons = dict.fromkeys(object_ids, "previous chunk is empty")
            else:
                first = int(rows.index.get_level_values("frame_idx").min())
                new_rles, new_ids = _frame_rles(rows, first)
                # The previous chunk's last frame whose objects overlap best
                prev_frames = sorted(prev.index.get_level_values("frame_idx").unique())
                best = None
                for frame_idx in prev_frames[-lookback_frames:]:
                    old_rles, old_ids = _frame_rles(prev, frame_idx)
                    if not new_rles or not old_rles:
                        continue
                    iou = mask_util.iou(new_rles, old_rles, [0] * len(old_rles))
                    pairs = list(zip(*linear_sum_assignment(iou, maximize=True)))
                    mean_iou = sum(iou[i, j] for i, j in pairs) / len(new_ids)
                    if best is None or mean_iou > best[0]:
                        best = (mean_iou, frame_idx, old_ids, iou, pairs)
                if best is not None:
                    mean_iou, frame_idx, old_ids, iou, pairs = best
                    logger.info(
                        f"{tag} IDs from frame {frame_idx} of chunk {chunk_idx - 1} "
                        f"(mean IoU per object {mean_iou:.2f})"
                    )
                    for i, j in pairs:
                        if iou[i, j] >= min_iou:
                            id_map[new_ids[i]] = old_ids[j]
                        else:
                            reasons[new_ids[i]] = (
                                f"IoU {iou[i, j]:.2f} < {min_iou} with "
                                f"object {old_ids[j]} in frame {frame_idx}"
                            )
            for oid in object_ids:
                if oid not in id_map:
                    id_map[oid] = next_id
                    logger.warning(
                        f"{tag} object {oid} -> new ID {next_id} ({reasons[oid]})"
                    )
                    next_id += 1
        linked[chunk_idx] = rows.rename(index=id_map, level="object_id").sort_index()
        next_id = max(next_id, max(id_map.values(), default=-1) + 1)
    return linked
