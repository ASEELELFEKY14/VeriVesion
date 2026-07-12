from dataclasses import dataclass

import cv2 as cv
import numpy as np
import torch

from .preprocessing import evaluate_frame_quality


@dataclass
class ExtractedFrame:
    frame_index: int
    timestamp_sec: float
    image_bgr: np.ndarray
    sharpness: float
    probability: float = None
    label: str = None
    predicted_class: int = None
    roi_rgb: np.ndarray = None
    heatmap_rgb: np.ndarray = None


try:
    import config

    SAMPLING_TIME_SEC = getattr(config, "SAMPLING_TIME_SEC", 5.0)
    MAX_FRAMES = getattr(config, "MAX_FRAMES", 60)
    MIN_FRAME_VARIANCE = getattr(config, "MIN_FRAME_VARIANCE", 25.0)
    MIN_BRIGHTNESS = getattr(config, "MIN_BRIGHTNESS", 30.0)
    MAX_BRIGHTNESS = getattr(config, "MAX_BRIGHTNESS", 245.0)
    CROP_FACTOR = getattr(config, "CROP_FACTOR", 0.85)
    QUALITY_SCAN_MULTIPLIER = getattr(config, "QUALITY_SCAN_MULTIPLIER", 4)

except (ImportError, NameError):
    SAMPLING_TIME_SEC = 5.0
    MAX_FRAMES = 60
    MIN_FRAME_VARIANCE = 25.0
    MIN_BRIGHTNESS = 30.0
    MAX_BRIGHTNESS = 245.0
    CROP_FACTOR = 0.85
    QUALITY_SCAN_MULTIPLIER = 4


SHARPNESS_THRESHOLD = 30.0


def get_roi_gray(
    image_bgr: np.ndarray,
    crop_factor: float = CROP_FACTOR
) -> np.ndarray:
    height, width = image_bgr.shape[:2]

    size = int(min(width, height) * crop_factor)

    left = (width - size) // 2
    top = (height - size) // 2

    cropped_bgr = image_bgr[top:top + size, left:left + size]

    return cv.cvtColor(cropped_bgr, cv.COLOR_BGR2GRAY)


def frame_sharpness(gray_roi: np.ndarray) -> float:
    return float(cv.Laplacian(gray_roi, cv.CV_64F).var())


def should_skip_frame(gray_roi: np.ndarray, strict: bool = True) -> bool:
    """
    Old helper kept for compatibility.
    The main video selection now uses evaluate_frame_quality from preprocessing.py.
    """
    total_pixels = gray_roi.size

    _, spotlight_mask = cv.threshold(
        gray_roi,
        245,
        255,
        cv.THRESH_BINARY
    )

    spotlight_ratio = np.sum(spotlight_mask == 255) / total_pixels

    max_spotlight_allowed = 0.20 if not strict else 0.06

    if spotlight_ratio > max_spotlight_allowed:
        return True

    _, dark_lumen_mask = cv.threshold(
        gray_roi,
        35,
        255,
        cv.THRESH_BINARY_INV
    )

    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        (15, 15)
    )

    dark_lumen_mask = cv.morphologyEx(
        dark_lumen_mask,
        cv.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv.findContours(
        dark_lumen_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    max_lumen_allowed = 0.30 if not strict else 0.12

    for contour in contours:
        area = cv.contourArea(contour)
        area_ratio = area / total_pixels

        if area_ratio > max_lumen_allowed:
            return True

    if np.std(gray_roi) < 5:
        return True

    return False


def _analyze_candidate(
    model,
    device,
    frame_bgr: np.ndarray,
    frame_count: int,
    fps: float,
    sharpness: float
) -> ExtractedFrame:
    from src.predictor import analyze_image

    analysis_result = analyze_image(model, device, frame_bgr)

    return ExtractedFrame(
        frame_index=frame_count,
        timestamp_sec=frame_count / fps,
        image_bgr=frame_bgr,
        sharpness=sharpness,
        probability=getattr(analysis_result, "probability", 0.5),
        label=getattr(analysis_result, "label", "Normal"),
        predicted_class=getattr(analysis_result, "predicted_class", 0),
        roi_rgb=getattr(analysis_result, "roi_rgb", None),
        heatmap_rgb=getattr(analysis_result, "heatmap_rgb", None),
    )


def extract_frames(
    video_path: str,
    model,
    device,
    top_k: int = 5
) -> dict | None:
    emergency_backup_frames = []
    all_filtered_frames = []

    frame_count = 0
    fps = 30.0
    final_mode = "strict"

    for attempt in ["strict", "relaxed"]:
        cap = cv.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Cannot open video file: {video_path}")
            return None

        fps = cap.get(cv.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30.0

        current_sampling = (
            SAMPLING_TIME_SEC
            if attempt == "strict"
            else min(2.0, SAMPLING_TIME_SEC)
        )

        frame_interval = max(1, int(fps * current_sampling))

        frame_count = 0
        candidates = []

        is_strict = attempt == "strict"

        candidate_limit = max(
            MAX_FRAMES * QUALITY_SCAN_MULTIPLIER,
            top_k
        )

        while cap.isOpened() and len(candidates) < candidate_limit:
            ret, frame_bgr = cap.read()

            if not ret:
                break

            if frame_count % frame_interval == 0:
                quality = evaluate_frame_quality(
                    frame_bgr,
                    strict=is_strict
                )

                if attempt == "strict":
                    emergency_backup_frames.append({
                        "frame_count": frame_count,
                        "frame_bgr": frame_bgr.copy(),
                        "sharpness": quality.sharpness,
                        "quality_score": quality.score,
                        "quality_reason": quality.reason,
                    })

                if quality.accepted:
                    candidates.append({
                        "frame_count": frame_count,
                        "frame_bgr": frame_bgr.copy(),
                        "quality": quality,
                    })

            frame_count += 1

        cap.release()

        if len(candidates) > 0:
            final_mode = attempt

            candidates.sort(
                key=lambda item: item["quality"].score,
                reverse=True
            )

            selected_candidates = candidates[:MAX_FRAMES]

            for item in selected_candidates:
                quality = item["quality"]

                extracted_frame = _analyze_candidate(
                    model=model,
                    device=device,
                    frame_bgr=item["frame_bgr"],
                    frame_count=item["frame_count"],
                    fps=fps,
                    sharpness=quality.sharpness,
                )

                all_filtered_frames.append(extracted_frame)

            break

    if len(all_filtered_frames) == 0:
        if len(emergency_backup_frames) == 0:
            print("Severe Error: Video file appears empty or corrupted.")
            return None

        print("Warning: No ideal diagnostic frames found. Using best available frames.")
        final_mode = "best_available"

        emergency_backup_frames.sort(
            key=lambda item: item.get("quality_score", item["sharpness"]),
            reverse=True
        )

        best_available = emergency_backup_frames[:top_k]

        for item in best_available:
            extracted_frame = _analyze_candidate(
                model=model,
                device=device,
                frame_bgr=item["frame_bgr"],
                frame_count=item["frame_count"],
                fps=fps,
                sharpness=item["sharpness"],
            )

            all_filtered_frames.append(extracted_frame)

    all_filtered_frames.sort(
        key=lambda frame: frame.probability,
        reverse=True
    )

    actual_k = min(len(all_filtered_frames), top_k)
    top_k_frames = all_filtered_frames[:actual_k]

    frame_predictions = [
        frame.probability
        for frame in all_filtered_frames
    ]

    top_k_probs = [
        frame.probability
        for frame in top_k_frames
    ]

    decision_score = float(np.mean(top_k_probs))
    final_decision = 1 if decision_score >= 0.5 else 0

    print("\n--- Analysis Complete ---")
    print(f"Frames Scanned: {frame_count}")
    print(f"Frames Selected and Analyzed: {actual_k} (Mode: {final_mode})")
    print(f"Video Decision Score: {decision_score:.4f}")

    return {
        "decision": final_decision,
        "decision_score": decision_score,
        "all_frame_probs": frame_predictions,
        "extracted_frames": top_k_frames,
    }