"""
IO utilities for loading video metadata and frames.
"""

import re
import sys

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import cv2

from loguru import logger

# Human-readable, chronologically sortable, e.g. 2026-10-06_18-28-51.37
RUN_ID_FORMAT = "%Y-%m-%d_%H-%M-%S.%f"


def get_run_id() -> str:
    """
    Get unique run identifier (current timestamp, to hundredths of a second).
    Run artefacts are named ``{run_id}_{artefact_basename}``.
    """
    # %f is microseconds (6 digits); truncate to hundredths (.ff)
    return datetime.now().strftime(RUN_ID_FORMAT)[:-4]


def setup_logger(
    log_dir: Path = Path("tmp/logs"),
    job_type: str = "sam3_hf",
    debug: bool = False,
    run_id: str | None = None,
) -> Path:
    """
    Configure loguru logger with both console and file output.

    The log file is ``{run_id}_{job_type}.log`` when *run_id* is given,
    else ``{job_type}_{timestamp}.log``.

    Returns:
        Path to the log file.
    """
    level = "DEBUG" if debug else "INFO"
    log_dir.mkdir(parents=True, exist_ok=True)

    if run_id is not None:
        log_filename = log_dir / f"{run_id}_{job_type}.log"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = log_dir / f"{job_type}_{timestamp}.log"

    logger.remove()

    # Console handler (colored)
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
        "[<level>{level}</level>] {message}",
        level=level,
    )

    # File handler
    logger.add(
        str(log_filename),
        format="{time:YYYY-MM-DD HH:mm:ss} [{level}] {message}",
        level=level,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    return log_filename


def create_run_directory(base_output_dir: Path, job_type: str) -> Path:
    """Create a timestamped run directory for this job."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / f"{timestamp}_{job_type}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def create_video_run_directory(base_output_dir: Path, video_stem: str) -> Path:
    """Create a dataset-aligned run directory: ``{base}/day_{N}/{sanitized_stem}/``.

    Parses the video ID from *video_stem* via
    :func:`~src.dataset.utils.extract_video_id` to obtain the day number.
    Falls back to a flat ``{base}/{sanitized_stem}/`` if parsing fails.
    """
    from src.dataset.utils import extract_video_id

    sanitized = sanitize_filename(video_stem)
    video_id = extract_video_id(video_stem)
    if video_id is not None:
        day = video_id[5:]  # "C1G3D28" → "28"
        run_dir = base_output_dir / f"day_{day}" / sanitized
    else:
        logger.warning(
            f"Could not parse video ID from '{video_stem}'; "
            "output will not be grouped by day"
        )
        run_dir = base_output_dir / sanitized
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def sanitize_filename(name: str) -> str:
    """Sanitize a stem string for use as a directory name."""
    sanitized = re.sub(r"[^\w\-]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_") or "video"


def get_video_metadata(video_path: str | Path) -> tuple[float, int]:
    """Return (fps, total_frames) for a video without loading all frames into RAM."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames


def get_downscaled_size(
    height: int, width: int, max_short_side: int = 1080
) -> tuple[int, int] | None:
    """(H, W) with the short side shrunk to *max_short_side*, keeping aspect ratio.

    E.g. 4K (2160, 3840) -> (1080, 1920). Sides are rounded to even numbers,
    which video encoders need. Returns ``None`` if no downscaling is needed.
    """
    scale = max_short_side / min(height, width)
    if scale >= 1:
        return None
    return 2 * round(height * scale / 2), 2 * round(width * scale / 2)


def load_transforms(size: tuple[int, int], antialias: bool = True) -> Callable:
    """Frame transforms for (N, C, H, W) uint8 batches: resize to *size* (H, W)."""
    from torchvision.transforms import v2

    return v2.Compose([v2.Resize(size=size, antialias=antialias)])


def get_torchcodec_decoder(video_path: str | Path, device: str = "cpu"):
    """Return the torchcodec decoder for *video_path*, building it on first use.

    The decoder is cached per video path and device so the index scan (~18 s
    on first access) only happens once. Its ``.metadata`` has the exact frame count,
    FPS and resolution, so no separate cv2 probe is needed.

    When *device* is ``"cuda"``, decoding is offloaded to NVDEC hardware; on
    the CPU, FFmpeg picks the number of decoding threads (about 3.5x faster
    than torchcodec's default of one for 4K HEVC).
    """
    from torchcodec.decoders import VideoDecoder

    key = (str(video_path), device)
    if key not in _torchcodec_decoder_cache:
        logger.info(
            "Building torchcodec index for '{}' (device={}).",
            Path(video_path).name,
            device,
        )
        threads = {"num_ffmpeg_threads": 0} if device == "cpu" else {}
        _torchcodec_decoder_cache[key] = VideoDecoder(
            str(video_path), device=device, **threads
        )
    return _torchcodec_decoder_cache[key]


def load_video_frames_torchcodec(
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    device: str = "cpu",
    transform: Callable | None = None,
    batch_size: int = 32,
) -> list:
    """Load frames [start_frame, end_frame) as a list of RGB numpy arrays.

    Uses torchcodec for frame-accurate decoding (seek_mode="exact"), via the
    cached decoder from :func:`get_torchcodec_decoder`.

    Frames are decoded *batch_size* at a time, and *transform* (e.g. from
    :func:`load_transforms`) is applied to each (N, C, H, W) batch on
    *device*, so full-size frames (e.g. 4K) never pile up in memory.
    """
    decoder = get_torchcodec_decoder(video_path, device=device)
    frames = []
    for start in range(start_frame, end_frame, batch_size):
        stop = min(start + batch_size, end_frame)
        # (N, C, H, W) uint8 tensor
        batch = decoder.get_frames_in_range(start=start, stop=stop).data
        if transform is not None:
            batch = transform(batch)
        frames.extend(frame.permute(1, 2, 0).cpu().numpy() for frame in batch)
    return frames


def load_video_frames_sequential(video_path, start_frame, end_frame):
    """Load frames [start_frame, end_frame) sequentially via cv2.

    Slower than seek-based loading but frame-accurate for all codecs
    (including H.264). Use this when torchcodec is not available (e.g.
    JAX environments).
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for i in range(end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        if i >= start_frame:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_video_frames_range(
    video_path: str | Path, start_frame: int, end_frame: int
) -> list:
    """Load frames [start_frame, end_frame) from a video file as a list of RGB numpy arrays.

    .. deprecated::
        Uses cv2.CAP_PROP_POS_FRAMES seeking, which is unreliable for
        H.264-encoded videos. Use :func:`load_video_frames` instead.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


_torchcodec_decoder_cache: dict[tuple[str, str], object] = {}

FRAME_LOADERS = {"torchcodec", "cv2_sequential", "cv2_seek"}


def load_video_frames(
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    loader: str = "torchcodec",
    device: str = "cpu",
    transform: Callable | None = None,
) -> list:
    """Load frames [start_frame, end_frame) using the specified loader.

    *transform* (e.g. from :func:`load_transforms`) is only supported by the
    ``"torchcodec"`` loader.
    """
    if loader == "torchcodec":
        return load_video_frames_torchcodec(
            video_path, start_frame, end_frame, device=device, transform=transform
        )
    if transform is not None:
        raise ValueError("transform is only supported by the torchcodec loader")
    if loader == "cv2_sequential":
        return load_video_frames_sequential(video_path, start_frame, end_frame)
    if loader == "cv2_seek":
        return load_video_frames_range(video_path, start_frame, end_frame)
    raise ValueError(f"Unknown loader {loader!r}; expected one of {FRAME_LOADERS}")
