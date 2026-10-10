# Tracking configs

Each YAML file configures one tracker variant. The config file stem determines
the output directory under `data/results/tracking/`.

## Variants

| Config                         | Variant  | Backend | Chunking               | Grounding             | Recovery                  | Description                                                                                   |
| ------------------------------ | -------- | ------- | ---------------------- | --------------------- | ------------------------- | --------------------------------------------------------------------------------------------- |
| `sam3_best.yaml`               | E        | SAM3    | Adaptive (YOLO-guided) | Adaptive (best-frame) | Fallback to prev chunk    | Full method.                                                                                  |
| `sam3_adaptive_grounding.yaml` | D        | SAM3    | Fixed 60 s             | Adaptive (best-frame) | Fallback to prev chunk    | Ablation: adaptive grounding without adaptive chunking.                                       |
| `sam3_baseline.yaml`           | C-strict | SAM3    | Fixed 60 s             | Frame-zero only       | None                      | Ablation: no adaptive grounding, no fallbacks.                                                |
| `gs2_adaptive_recovery.yaml`   | B-parity | GS2     | Fixed 60 s             | GDINO per chunk       | Re-init on carryover loss | GS2 with recovery to match SAM3 scaffolding.                                                  |
| `gs2_baseline.yaml`            | B-strict | GS2     | Fixed 60 s             | GDINO per chunk       | None                      | GS2 baseline, no recovery mechanisms.                                                         |
| `yolo_botsort.yaml`            | A        | YOLO    | Adaptive (YOLO-guided) | N/A                   | N/A                       | YOLO + BoT-SORT only (no SAM3). Not run separately — results extracted from `sam3_best` runs. |
| `yolo_botsort_reid_on.yaml`    | A1       | YOLO    | Adaptive (YOLO-guided) | N/A                   | N/A                       | YOLO + BoT-SORT with ReID enabled.                                                            |

## Other files

| File                   | Description                                                     |
| ---------------------- | --------------------------------------------------------------- |
| `benchmark_videos.txt` | Stems of the 5 benchmark videos used in `--eval` mode.          |
| `tracker_simple.yaml`  | Default settings for `pipeline/run_tracker_simple.py`.          |
| `ultralytics/`         | Ultralytics tracker configs (BoT-SORT parameters, ReID toggle). |
