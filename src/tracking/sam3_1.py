"""SAM 3.1 (facebookresearch/sam3, "Object Multiplex") grounding and tracking.

Used chunk by chunk by ``pipeline/run_tracker_simple.py``:

- ``Sam31.ground``: the text-prompted SAM 3.1 video predictor (detector and
  tracker) on the grounding frames. Same signature and output dicts as
  ``sam3._process_video_chunk``. Like SAM 3, it holds new objects back until
  they are confirmed, so the first ~10 frames come out empty.
- ``Sam31.track``: the SAM 3.1 multiplex tracker on its own, seeded on one
  frame with click points or masks and run forwards and backwards from it.
  Output dicts match ``sam3._process_tracker_chunk``. All objects share one
  multiplex state, so they are tracked jointly. Clicks are first turned into
  masks one object at a time, as the SAM 3.1 predictor does: clicking several
  objects into one multiplex state gives much worse masks.

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

from pathlib import Path

import numpy as np
import torch

from huggingface_hub import hf_hub_download
from loguru import logger
from PIL import Image

from .utils import FrameTensors, masks_to_frame_output

SAM3_DIR = Path(__file__).resolve().parents[2] / "ext" / "sam3"
SAM31_REPO = "facebook/sam3.1"
SAM31_CHECKPOINT = "sam3.1_multiplex.pt"
# Upper bound on objects the text-prompted predictor tracks (upstream: 16)
MAX_OBJECTS = 64
# SAM 3.1 input normalisation
IMAGE_MEAN = IMAGE_STD = (0.5, 0.5, 0.5)


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

    Building the predictor turns on bf16 autocast for the rest of the process
    (upstream behaviour); ``close`` turns it off again.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        self._predictor = None
        self._tracker = None

    @property
    def predictor(self):
        """Text-prompted video predictor (detector + tracker)."""
        if self._predictor is None:
            logger.info("Building SAM 3.1 video predictor")
            predictor = _model_builder().build_sam3_multiplex_video_predictor(
                use_fa3=False,  # FlashAttention 3 needs a Hopper GPU
                max_num_objects=MAX_OBJECTS,
            )
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
                load_from_HF=False, use_fa3=False, device=self.device
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
            tracker.load_state_dict(state_dict, strict=True)
            tracker.eval()
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
        Returns {global frame index: output dict} for every frame of the chunk.
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
                    )
        del state
        return outputs_per_frame

    def close(self) -> None:
        """Free the models and undo the predictor's global bf16 autocast."""
        if self._predictor is not None:
            self._predictor.bf16_context.__exit__(None, None, None)
            self._predictor.shutdown()
        self._predictor = self._tracker = None
