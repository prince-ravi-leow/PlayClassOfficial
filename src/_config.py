from typing import Final

DEFAULT_CONFIG_DIR: Final = "config"
DEFAULT_DATA_DIR: Final = "data"
DEFAULT_DATASET_DIR: Final = f"{DEFAULT_DATA_DIR}/dataset"
DEFAULT_CHECKPOINT_DIR: Final = f"{DEFAULT_DATA_DIR}/eval"
DEFAULT_LABEL_DIR: Final = f"{DEFAULT_DATA_DIR}/labels"
DEFAULT_POSTPROCESSING_DIR: Final = f"{DEFAULT_DATA_DIR}/postprocessing"
DEFAULT_RESULTS_DIR: Final = f"{DEFAULT_DATA_DIR}/results"
DEFAULT_CLUSTERING_DIR: Final = f"{DEFAULT_RESULTS_DIR}/clustering"
DEFAULT_TRACKING_DIR: Final = f"{DEFAULT_RESULTS_DIR}/tracking"
DEFAULT_VIDEO_DIR: Final = f"{DEFAULT_DATA_DIR}/videos"
DEFAULT_TRACKER_CONFIG: Final = f"{DEFAULT_CONFIG_DIR}/ultralytics/botsort.yaml"
# Discard windows where less than this fraction of frames are available after postprocessing
DEFAULT_MIN_WINDOW_COVERAGE: Final = 0.5
DEFAULT_N_JOBS: Final = 10
DEFAULT_FPS: Final = 25.0
DEFAULT_N_BIRDS: Final = 3
K_RANGE: Final[range] = range(2, 13)
LABEL_ORDER: Final = ["none", "worm", "locomotor", "social"]
RANDOM_SEED: Final[int] = 42
