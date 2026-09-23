# Tracking outputs

Each subdirectory corresponds to one tracker config (`config/{stem}.yaml`).
Videos are grouped by day (`day_28/`, `day_29/`) in production runs, or stored
flat when only benchmark videos were tracked (see `--eval` flag).

## Variants

| Directory                   | Variant  | Backend | Notes                                                                                                                                  |
| --------------------------- | -------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `SAM 3_best/`               | E        | SAM 3   | Full method (adaptive chunking + adaptive grounding). Also contains YOLO BoT-SORT results (`yolo_tracking.parquet`) used as variant A. |
| `SAM 3_adaptive_grounding/` | D        | SAM 3   | Adaptive grounding, fixed 60 s chunks.                                                                                                 |
| `SAM 3_baseline/`           | C-strict | SAM 3   | Frame-0 grounding only, no fallbacks.                                                                                                  |
| `gs2_adaptive_recovery/`    | B-parity | GS2     | GS2 with recovery, matching SAM 3 scaffolding.                                                                                         |
| `gs2_baseline/`             | B-strict | GS2     | GS2 fixed 60 s, no recovery. Incomplete: `C3G2` failed during tracking.                                                                |
| `yolo_botsort_reid_on/`     | A1       | YOLO    | BoT-SORT with ReID enabled.                                                                                                            |
| `yolo_botsort/`             | A        | YOLO    | BoT-SORT without ReID. Not produced separately — variant A results are extracted from `SAM 3_best/` runs.                              |

## Per-video output

A completed run directory contains:

- `yolo_tracking.parquet` — YOLO + BoT-SORT detections/tracks
- `tracking_outputs.parquet` — SAM 3/GS2 mask-level tracks (not present for
  YOLO-only runs)
- `chunk_info.json` — adaptive chunking metadata
- `metrics/` — per-chunk tracking metrics
- Log file (`*.log`) and a copy of the config YAML
