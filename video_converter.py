import shutil
import subprocess
from pathlib import Path
import cv2

SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", 
    ".webm", ".mpeg", ".mpg", ".3gp", ".m4v"
}

def _resolve_ffmpeg() -> str | None:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None

def _ffmpeg_available() -> bool:
    return _resolve_ffmpeg() is not None

def convert_video_to_mp4(source_path: str | Path, target_path: str | Path) -> Path:
    source_path = Path(source_path)
    target_path = Path(target_path)
    
    ffmpeg_exe = _resolve_ffmpeg()
    if not ffmpeg_exe:
        raise RuntimeError("ffmpeg is not available for video conversion.")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_exe,
        "-y",
        "-i", str(source_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(target_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Video conversion failed: {result.stderr.strip() or result.stdout.strip()}")
    return target_path

def _can_open_with_opencv(video_path: Path) -> bool:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return False
    success, _ = capture.read()
    capture.release()
    return success

def ensure_compatible_video(source_path: str | Path, output_dir: Path) -> tuple[Path, bool]:
    
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Video file not found: {source_path}")

    suffix = source_path.suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video format '{suffix}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}"
        )

    if _can_open_with_opencv(source_path) and suffix == ".mp4":
        return source_path, False

    if not _ffmpeg_available():
        raise RuntimeError(
            "This video format cannot be read directly. Install ffmpeg and try again."
        )

    
    converted_folder = output_dir / "converted"
    converted_folder.mkdir(parents=True, exist_ok=True)
    
    converted_path = converted_folder / f"{source_path.stem}_converted.mp4"
    convert_video_to_mp4(source_path, converted_path)

    if not _can_open_with_opencv(converted_path):
        raise RuntimeError("OpenCV failed to read the video even after successful conversion.")

    return converted_path, True