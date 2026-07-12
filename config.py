from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Paths
MODEL_PATH = PROJECT_ROOT / "models" / "res_18_model_layers(4,fc).pth"
VIDEO_DATASET_PATH = Path(r"C:\Users\CM\OneDrive\Desktop\Videos dataset")
OUTPUT_DIR = PROJECT_ROOT / "outputs"
NOTEBOOK_PATH = Path(r"C:\Users\CM\Downloads\_Res18_with_cropping_ROI_(layers).ipynb")

# Model settings
IMG_SIZE = 224
HIDDEN_SIZE = 347
DROPOUT = 0.5

POSITIVE_CLASS = 1
NEGATIVE_CLASS = 0

CLASS_LABELS = {
    0: "Negative Inflammation",
    1: "Positive Inflammation",
}

THRESHOLD = 0.5

# ROI crop
CROP_FACTOR = 0.85

# Image upload settings
ALLOW_MULTIPLE_IMAGES = True
MAX_IMAGE_UPLOAD_COUNT = 20

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".webp",
}

STREAMLIT_IMAGE_TYPES = [
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "tiff",
    "webp",
]

# Video frame extraction & filtering
SAMPLING_TIME_SEC = 5.0
MAX_FRAMES = 60
MIN_FRAME_VARIANCE = 25.0

MIN_BRIGHTNESS = 30.0
MAX_BRIGHTNESS = 240.0

# Extra video quality filtering
MIN_FRAME_SHARPNESS = 20.0
MIN_FRAME_CONTRAST = 18.0
MIN_TISSUE_RATIO = 0.25
MAX_SPECULAR_RATIO = 0.15
MAX_DARK_RATIO = 0.45
MAX_LARGE_DARK_REGION_RATIO = 0.18

# The extractor scans more frames than it finally analyzes, then keeps the best quality frames.
QUALITY_SCAN_MULTIPLIER = 4

# Supported video formats
SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".webm",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".3gp",
    ".flv",
}

# Report
TOP_FRAMES_COUNT = 5
EXTRACTED_FRAMES_DISPLAY_COUNT = 8

DISCLAIMER = (
    "This system is a decision-support tool for endoscopy video analysis. "
    "It does not replace professional medical diagnosis by a qualified specialist."
)