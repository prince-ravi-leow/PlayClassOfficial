"""Project-wide constants: default paths, dataset settings, and classification/clustering defaults."""

from typing import Final

DEFAULT_CONFIG_DIR: Final = "config"
DEFAULT_DATA_DIR: Final = "data"
DEFAULT_DATASET_DIR: Final = f"{DEFAULT_DATA_DIR}/dataset"
DEFAULT_LABEL_DIR: Final = f"{DEFAULT_DATA_DIR}/labels"
DEFAULT_POSTPROCESSING_DIR: Final = f"{DEFAULT_DATA_DIR}/postprocessing"
DEFAULT_RESULTS_DIR: Final = f"{DEFAULT_DATA_DIR}/results"
DEFAULT_CHECKPOINT_DIR: Final = f"{DEFAULT_RESULTS_DIR}/eval_classification"
DEFAULT_CLUSTERING_DIR: Final = f"{DEFAULT_RESULTS_DIR}/clustering"
DEFAULT_TRACKING_DIR: Final = f"{DEFAULT_RESULTS_DIR}/tracking"
DEFAULT_VIDEO_DIR: Final = f"{DEFAULT_DATA_DIR}/videos"
DEFAULT_TRACKER_CONFIG: Final = f"{DEFAULT_CONFIG_DIR}/ultralytics/botsort.yaml"
# Discard windows where less than this fraction of frames are available after postprocessing
DEFAULT_MIN_WINDOW_COVERAGE: Final = 0.5
DEFAULT_N_JOBS: Final = 10
DEFAULT_FPS: Final = 25.0
DEFAULT_N_BIRDS: Final = 3
LABEL_ORDER: Final = ["none", "worm", "locomotor", "social"]
K_RANGE: Final = range(2, 13)
RANDOM_SEED: Final = 42
