# Tracking

Multi-object tracking pipeline supporting SAM 3, Grounded-SAM-2, and YOLO
backends. Each tracker variant has its own YAML config in `config/` — see
[`config/README.md`](../config/README.md) for the full variant table.

> [!IMPORTANT] Set `video_path` or `video_dir` in your config before running.
> Requires CUDA GPU.

## Commands

```sh
# Run the default variant (sam3_best)
pixi run -e tracker track_best

# Run with a specific config
pixi run -e tracker python -m pipeline.run_tracker --config config/sam3_baseline.yaml

# Run all variants
pixi run track_all

# Run a subset (yolo / sam3 / gs2)
pixi run track_sam3

# Benchmark mode — track only the 5 eval videos (listed in config/benchmark_videos.txt)
pixi run eval_all
pixi run eval_sam3 --overwrite   # re-track even if output exists

# Test that SAM 3 loads and runs on a short clip
pixi run -e tracker test_tracker
```

GS2 variants run in the `gs2` environment; all others use `tracker`. The
`track_all` / `eval_all` scripts handle this automatically.

## Output layout

```
data/results/tracking/{config_stem}/
└── day_{N}/{video_stem}/
    ├── {config_stem}.yaml              # config copy
    ├── tracking_outputs.parquet        # per-frame tracks (frame_idx, object_id)
    ├── chunk_info.json                 # per-chunk metadata + prompt points
    ├── yolo_tracking.parquet            # raw YOLO scan detections
    ├── metrics/
    │   ├── per_frame_metrics.parquet
    │   ├── per_id_metrics.parquet
    │   ├── summary_metrics.parquet
    │   ├── yolo_scan_metrics.parquet   # per-frame occlusion/separation scores
    │   └── yolo_scan_summary.parquet
    └── visualizations/
        ├── id_timeline.png
        ├── dashboard.png
        ├── yolo_scan_overview.png
        └── chunk_boundaries.png
```

## Chunking modes (SAM 3)

| Mode         | Config key                                 | Description                                                                                              |
| ------------ | ------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| **Adaptive** | `use_adaptive_chunking: true`              | YOLO pre-scan finds high-separation windows; boundaries placed within ±`adaptive_search_window_seconds`. |
| **Fixed**    | `use_adaptive_chunking: false`             | Uniform segments of `chunk_seconds`.                                                                     |
| **Manual**   | `manual_chunk_frames: [[0,375], ...]`      | Explicit `[start, end]` frame pairs.                                                                     |
| **Reuse**    | `reuse_chunk_info: true` + `reuse_run_dir` | Load boundaries from a previous run.                                                                     |

## Grounding (SAM 3)

At each chunk boundary, a text prompt (`text_prompt: "bird"`) searches
`grounding_frames` frames for the best initialisation frame. IDs are matched
across chunks by mask IoU (`id_match_iou_threshold`). If grounding fails,
falls back to the previous chunk's masks (`fallback_to_prev_chunk`).

Optional keys (off by default in every config):

| Key                                              | Effect                                                                                                                              |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `text_grounding.fill_grounding_gap: true`        | Fill the frames before the chosen grounding frame with the grounding model's own outputs. The day-28 production runs used this.    |
| `text_grounding.best_overlap_id_matching: true`  | Pick the previous-chunk reference frame by mask overlap instead of the latest frame with enough objects. Legacy; no production run. |
| `text_grounding.id_match_min_ratio` (0.5)        | Chunks whose ID match ratio falls below this are flagged with `id_discontinuity` in `chunk_info.json`.                              |
| `frame_loader` (`torchcodec`)                    | Frame decoder: `torchcodec`, `cv2_seek` or `cv2_sequential`. The day-28 runs predate torchcodec and need `cv2_seek` to reproduce.   |

## Recomputing chunk boundaries

Recompute YOLO scan metrics and chunk boundaries from an existing run without
re-running SAM 3:

```sh
pixi run -e tracker python -m pipeline.compute_chunk_boundaries \
    --run-dir data/results/tracking/{config_stem}/day_{N}/{video_stem} \
    --video-dir data/videos/day_{N}
```
