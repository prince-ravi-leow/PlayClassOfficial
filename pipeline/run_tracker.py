"""
Unified launcher for tracking pipelines (SAM 3, Grounded-SAM-2).

Each video is processed in chunks: chunk 0 uses Sam3VideoModel (text-prompted
segmentation); subsequent chunks use Sam3TrackerVideoModel (point-prompted),
initialised from masks sampled at the cleanest frame in the previous chunk.

Usage:
    # Run with default config (config/sam3_best.yaml)
    pixi run -e tracker track_best
    # Run with custom config
    pixi run -e tracker python -m pipeline.run_tracker --config config/sam3_baseline.yaml
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_ALLOC_CONF"] = (
    "expandable_segments:True,garbage_collection_threshold:0.6"
)

from argparse import ArgumentParser
from pathlib import Path

from omegaconf import OmegaConf

from src._config import DEFAULT_TRACKING_DIR, DEFAULT_VIDEO_DIR
from src.utils.io import create_video_run_directory

DEFAULT_CONFIG = "config/sam3_best.yaml"
BENCHMARK_VIDEOS = "config/benchmark_videos.txt"

VIDEO_EXTENSIONS = {
    ext for e in [".mp4", ".avi", ".mov", ".mkv", ".m4v"] for ext in (e, e.upper())
}


def _is_completed(run_dir: Path) -> bool:
    """A run directory is complete if it contains at least one .parquet file."""
    return run_dir.is_dir() and any(run_dir.glob("*.parquet"))


def _infer_backend(cfg) -> str:
    if "gs2" in cfg:
        return "gs2"
    return "sam3"


def _get_backend_module(backend: str):
    if backend == "sam3":
        from src.tracking.sam3 import _run_batch, _run_single_video
    elif backend == "gs2":
        from src.tracking.grounded_sam_2 import _run_batch, _run_single_video
    else:
        raise ValueError(f"Unknown backend: {backend!r}")
    return _run_batch, _run_single_video


def _load_benchmark_stems() -> set[str]:
    path = Path(BENCHMARK_VIDEOS)
    stems = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            stems.add(line)
    return stems


def _find_benchmark_videos(video_dir: Path, stems: set[str]) -> list[Path]:
    """Find benchmark video files under video_dir (recursive)."""
    found = []
    for f in sorted(video_dir.rglob("*")):
        if f.is_file() and f.suffix in VIDEO_EXTENSIONS and f.stem in stems:
            found.append(f)
    return found


def _run_single(
    cfg, video_file: Path, batch_dir: Path, config_path: Path, overwrite: bool
) -> None:
    run_dir = create_video_run_directory(batch_dir, video_file.stem)
    if not overwrite and _is_completed(run_dir):
        print(f"Skipping {video_file.stem} (output exists in {run_dir})")
        return
    _, _run_single_video = _get_backend_module(_infer_backend(cfg))
    _run_single_video(cfg, run_dir, config_path=config_path)


def run(
    cfg,
    config_path: str | Path,
    backend: str,
    video_path: str | Path,
    eval: bool = False,
    overwrite: bool = False,
) -> None:
    config_path = Path(config_path)
    video_path = Path(video_path)

    batch_dir = Path(DEFAULT_TRACKING_DIR) / config_path.stem
    batch_dir.mkdir(parents=True, exist_ok=True)

    if eval:
        stems = _load_benchmark_stems()
        if video_path.is_file():
            videos = [video_path] if video_path.stem in stems else []
        elif video_path.is_dir():
            videos = _find_benchmark_videos(video_path, stems)
        else:
            raise FileNotFoundError(f"Video path does not exist: {video_path}")
        if not videos:
            raise ValueError(f"No benchmark videos found under {video_path}")
        for video_file in videos:
            _run_single(cfg, video_file, batch_dir, config_path, overwrite)
    elif video_path.is_file():
        _run_single(cfg, video_path, batch_dir, config_path, overwrite)
    elif video_path.is_dir():
        if overwrite:
            _run_batch, _ = _get_backend_module(backend)
            _run_batch(cfg, batch_dir, video_path, config_path=config_path)
        else:
            videos = sorted(
                f
                for f in video_path.iterdir()
                if f.is_file() and f.suffix in VIDEO_EXTENSIONS
            )
            for video_file in videos:
                _run_single(cfg, video_file, batch_dir, config_path, overwrite)
    else:
        raise FileNotFoundError(f"Video path does not exist: {video_path}")


def main():
    parser = ArgumentParser(description="Tracking pipeline launcher")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to tracking config file.",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=DEFAULT_VIDEO_DIR,
        help="Video file or directory of videos.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Eval mode: track only benchmark videos listed in config/benchmark_videos.txt.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-track videos even if output already exists.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    backend = _infer_backend(cfg)

    run(
        cfg,
        config_path=args.config,
        backend=backend,
        video_path=args.video_path,
        eval=args.eval,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
