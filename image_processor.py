import cv2 as cv
import numpy as np

def image_sharpness(gray_image: np.ndarray) -> float:
    
    return float(cv.Laplacian(gray_image, cv.CV_64F).var())

def remove_vignetting(frame_bgr: np.ndarray, roi_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    
    h, w = frame_bgr.shape[:2]
    crop_factor = 0.85
    size = int(min(w, h) * crop_factor)

    left = (w - size) // 2
    top = (h - size) // 2
    
    img_cropped = frame_bgr[top:top+size, left:left+size]
    return cv.resize(img_cropped, roi_size, interpolation=cv.INTER_AREA)

def detect_and_crop_viewport(image: np.ndarray, buffer: int = 5) -> np.ndarray:
    
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(c)
        y_start, y_end = max(0, y - buffer), min(image.shape[0], y + h + buffer)
        x_start, x_end = max(0, x - buffer), min(image.shape[1], x + w + buffer)
        return image[y_start:y_end, x_start:x_end]
    
    return image

def validate_image(image_bgr: np.ndarray, min_variance: float) -> tuple[bool, str]:
    
    if image_bgr is None or image_bgr.size == 0:
        return False, "Invalid or empty image file."

    
    gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    
    
    variance = image_sharpness(gray)
    if variance < min_variance:
        return False, f"Image fails sharpness threshold (Score: {variance:.1f})."

    
    mean_brightness = float(gray.mean())
    if mean_brightness < 40.0 or mean_brightness > 200.0:
        return False, f"Image fails brightness bounds (Score: {mean_brightness:.1f})."

    return True, "Valid"

def process_raw_image(image_bgr: np.ndarray, buffer: int = 5) -> np.ndarray:
    
    cropped_viewport = detect_and_crop_viewport(image_bgr, buffer)
    processed_bgr = remove_vignetting(cropped_viewport)
    return processed_bgr