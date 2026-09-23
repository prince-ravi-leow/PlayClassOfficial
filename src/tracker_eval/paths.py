"""Single source of truth for all paths used by the tracker-eval pipeline.

Small artefacts (video manifest, keyframe schedule, results CSVs) live under
`data/results/eval_tracking/`; heavy ones (CVAT backup, ground-truth and
prediction MOT files) under its `tracker_benchmark/` subdir. Tracker run
outputs are read from `data/results/tracking/{config_stem}/`, and the
per-variant YAML configs live in `config/`.

Each subcommand's CLI accepts `--manifest` / `--out` / `--predictions-root`
overrides; these constants supply the defaults.
"""

from pathlib import Path
from typing import Final

from src._config import DEFAULT_RESULTS_DIR, DEFAULT_TRACKING_DIR, DEFAULT_VIDEO_DIR

ROOT: Final = Path(__file__).resolve().parents[2]

EVAL_TRACKING_DIR: Final = f"{DEFAULT_RESULTS_DIR}/eval_tracking"

MANIFEST_CSV: Final = f"{EVAL_TRACKING_DIR}/video_manifest.csv"
ANNOTATION_FRAMES: Final = f"{EVAL_TRACKING_DIR}/annotation_frames.csv"
RESULTS_DIR: Final = f"{EVAL_TRACKING_DIR}/results"

BENCHMARK_DIR: Final = f"{EVAL_TRACKING_DIR}/tracker_benchmark"
CVAT_BACKUP_DIR: Final = f"{BENCHMARK_DIR}/cvat_backup/playclass-tracker-eval"
GROUND_TRUTH_DIR: Final = f"{BENCHMARK_DIR}/ground_truth"
PREDICTIONS_MOT_DIR: Final = f"{BENCHMARK_DIR}/predictions_mot"

TRACKEVAL_DIR: Final = "ext/TrackEval"

DEFAULT_TRACKER_CONFIG: Final = "config/sam3_best.yaml"

# Per-variant tracker run directories (keyed by config stem)
TRACKER_RUNS_SAM3_BEST: Final = f"{DEFAULT_TRACKING_DIR}/sam3_best"
TRACKER_RUNS_SAM3_ADAPTIVE_GROUNDING: Final = (
    f"{DEFAULT_TRACKING_DIR}/sam3_adaptive_grounding"
)
TRACKER_RUNS_SAM3_BASELINE: Final = f"{DEFAULT_TRACKING_DIR}/sam3_baseline"
TRACKER_RUNS_GS2_BASELINE: Final = f"{DEFAULT_TRACKING_DIR}/gs2_baseline"
TRACKER_RUNS_GS2_ADAPTIVE_RECOVERY: Final = (
    f"{DEFAULT_TRACKING_DIR}/gs2_adaptive_recovery"
)
TRACKER_RUNS_YOLO_BOTSORT: Final = f"{DEFAULT_TRACKING_DIR}/yolo_botsort"
TRACKER_RUNS_YOLO_BOTSORT_REID_ON: Final = (
    f"{DEFAULT_TRACKING_DIR}/yolo_botsort_reid_on"
)

# build-manifest inputs: YOLO scans (`day_{N}/{video_stem}/`) + source videos
SCAN_RUNS_ROOT: Final = TRACKER_RUNS_SAM3_BEST
RAW_VIDEO_ROOT: Final = DEFAULT_VIDEO_DIR
