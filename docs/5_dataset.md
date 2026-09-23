# Dataset Build

| Script                         | Description                                                          |
| ------------------------------ | -------------------------------------------------------------------- |
| `pipeline/build_dataset.py`    | Postprocess tracking outputs, match bird IDs, build dataset parquets |
| `pipeline/extract_features.py` | Extract mask features + window summaries from dataset tracks (CPU)   |

---

## Overview

The dataset is built in three steps from two sources: the SAM3 tracking outputs
(`tracking_outputs.parquet` per video) and the registration protocol Excel files
(behaviour labels + bird identity).

**Step 1** (`build_dataset`) is lightweight and produces the canonical
`tracks.parquet` and `labels.parquet`. Steps 2 and 3 are slow and independent of
each other — run them in parallel if you have the resources.

---

## Step 1 — Labels, postprocessing, windows

```sh
pixi run build_dataset
```

**What it does:**

1. **Parses labels** from the registration Excel files — behaviour annotations
   (locomotor play, object play, no-play) timestamped per bird per video.
2. **Applies postprocessing** — reads each video's
   `tracking_postprocessing.json`, trims bad frame ranges, merges split tracks
   (`id_switch`), and renames tracker IDs to protocol bird IDs (`id_match`). See
   [postprocessing](4_postprocessing.md) for how to fill these in.
3. **Assigns temporal windows** — each behaviour label defines a fixed-duration
   window; track frames are assigned to the window they fall in.
4. **Filters incomplete windows** — windows where too few frames remain after
   trimming (default: <50% of expected frames) are dropped from both tracks and
   labels to avoid degenerate feature statistics.

**Output:** `data/dataset/tracks.parquet` + `data/dataset/labels.parquet`, keyed
by `(video_id, bird_id, window)`.

---

## Step 2 — Mask features (CPU)

```sh
pixi run extract_features
```

**What it does:** Decodes the RLE masks in `tracks.parquet` frame-by-frame and
computes 19 morphokinematic descriptors per bird per frame:

| Descriptor             | What it captures                                        |
| ---------------------- | ------------------------------------------------------- |
| **Shape**              |                                                         |
| `mask_area`            | Segmentation mask area (px²)                            |
| `aspect_ratio`         | Bounding-box width / height                             |
| `elongation`           | Major-to-minor axis ratio                               |
| `orientation`          | Major-axis angle (rad)                                  |
| `solidity`             | Mask area / convex-hull area                            |
| `eccentricity`         | Roundness to elongation (0–1)                           |
| `perimeter`            | Outer contour length (px)                               |
| `circularity`          | 4π × area / perimeter²                                  |
| **Motion**             |                                                         |
| `velocity`             | Centroid displacement between frames (px)               |
| `acceleration`         | Frame-to-frame change in velocity                       |
| `turning_angle`        | Angle between consecutive movement directions (rad)     |
| `velocity_autocorr`    | Product of consecutive frame speeds (px²)               |
| `area_change_rate`     | Frame-to-frame relative change in mask area             |
| `orientation_velocity` | Frame-to-frame change in orientation (rad)              |
| `solidity_change`      | Frame-to-frame change in solidity                       |
| `elongation_change`    | Frame-to-frame change in elongation                     |
| **Social**             |                                                         |
| `min_dist_to_other`    | Min centroid distance to other birds in the frame (px)  |
| `mean_dist_to_other`   | Mean centroid distance to other birds in the frame (px) |
| `dist_change_rate`     | Frame-to-frame change in nearest-bird distance (px)     |

These are then summarized per window (mean, std, median, MAD, skew, kurtosis,
CV, q10, q90 → `features_windowed.parquet`).

---

## Step 3 — Embeddings (GPU)

```sh
# DINOv3 ViT-L (default)
pixi run extract_dinov3

# V-JEPA 2.1 ViT-L temporal
pixi run extract_vjepa2 --temporal

# VideoPrism Base temporal
pixi run -e videoprism extract_videoprism --temporal
```

**What it does:** For each `(video_id, bird_id, window)`, crops the bird's
bounding box from video frames and runs them through a pretrained visual
backbone. The result is a per-window embedding tensor of shape
`(frames_in_window, D)`.

These capture appearance and motion information the handcrafted features miss —
coat colour, posture details, fine-grained motion texture.

Outputs are saved as `.pt` dicts keyed by `(video_id, bird_id, window)`, e.g.
`embeddings_dinov3_vitl.pt`.

---

## Output files

| File                        | Keyed by                         | Contents                                               |
| --------------------------- | -------------------------------- | ------------------------------------------------------ |
| `tracks.parquet`            | `(video_id, bird_id, frame_idx)` | Cleaned tracks with RLE masks, bbox, window assignment |
| `labels.parquet`            | `(video_id, bird_id, window)`    | Behaviour labels aligned to track coverage             |
| `features_all.parquet`      | `(video_id, bird_id, frame_idx)` | Per-frame mask features                                |
| `features_windowed.parquet` | `(video_id, bird_id, window)`    | Per-window feature summaries                           |
| `features_binned.pt`        | `(video_id, bird_id, window)`    | Temporal feature tensors (same format as embeddings)   |
| `embeddings_*.pt`           | `(video_id, bird_id, window)`    | Visual backbone embeddings, shape `(F_w, D)`           |
