"""
VisionPlate Utility Package - Image Utilities
"""
import os
import cv2
import numpy as np
from datetime import datetime
def cv2_to_qpixmap(cv_img: np.ndarray):
    """
    Convert OpenCV BGR image array to PySide6 QPixmap (Desktop GUI Helper).
    """
    try:
        from PySide6.QtGui import QImage, QPixmap
    except ImportError:
        return None

    if cv_img is None or cv_img.size == 0:
        return QPixmap()

    # If image is grayscale (2D)
    if len(cv_img.shape) == 2:
        h, w = cv_img.shape
        qimg = QImage(cv_img.data, w, h, w, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(qimg)

    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_img.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def crop_bbox(image: np.ndarray, bbox: list) -> np.ndarray:
    """
    Crop bounding box [x1, y1, x2, y2] safely from image numpy array.
    """
    if image is None or len(bbox) != 4:
        return np.zeros((10, 10, 3), dtype=np.uint8)

    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    # Safe bounds clipping
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    return cropped


def save_crop_to_disk(crop_img: np.ndarray, output_dir: str = "outputs/crops", prefix: str = "plate") -> str:
    """
    Saves cropped plate image to disk without overwriting existing files unnecessarily.
    Returns the saved file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)

    # Save using OpenCV
    cv2.imwrite(filepath, crop_img)
    return filepath
