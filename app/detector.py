"""
VisionPlate YOLO Detector Module
Handles loading trained YOLO checkpoint (best.pt) and performing object detection.
"""
import os
import cv2
import torch
import numpy as np
from typing import Dict, Any, List, Union, Tuple
from ultralytics import YOLO

from app.utils.image_utils import crop_bbox


class PlateDetector:
    """
    Number Plate Detector using pre-trained YOLO checkpoint (best.pt).
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            # Resolve relative to project root
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "best.pt")

        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.class_names = {0: "number_plate"}
        self.load_error = ""

    def load_model(self) -> Tuple[bool, str]:
        """
        Load YOLO model from model_path safely.
        """
        if not os.path.exists(self.model_path):
            self.is_loaded = False
            self.load_error = f"Model file not found at: '{self.model_path}'"
            return False, self.load_error

        try:
            # Load Ultralytics YOLO model
            self.model = YOLO(self.model_path)
            self.is_loaded = True
            if hasattr(self.model, "names") and self.model.names:
                self.class_names = self.model.names
            self.load_error = ""
            return True, "Model loaded successfully."
        except Exception as e:
            self.is_loaded = False
            self.load_error = f"Failed to load model: {str(e)}"
            return False, self.load_error

    def detect(self, image_input: Union[str, np.ndarray], conf_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Run YOLO detection on an image path or numpy BGR array.

        Returns:
            dict containing:
            - success (bool)
            - error (str)
            - original_image (np.ndarray BGR)
            - annotated_image (np.ndarray BGR)
            - detections (list of dicts with bbox, conf, class_name, crop)
        """
        if not self.is_loaded:
            success, msg = self.load_model()
            if not success:
                return {
                    "success": False,
                    "error": msg,
                    "original_image": None,
                    "annotated_image": None,
                    "detections": []
                }

        # Load image array if path passed
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                return {
                    "success": False,
                    "error": f"Image file not found: '{image_input}'",
                    "original_image": None,
                    "annotated_image": None,
                    "detections": []
                }
            image = cv2.imread(image_input)
            if image is None:
                return {
                    "success": False,
                    "error": f"Unable to decode image file: '{image_input}'",
                    "original_image": None,
                    "annotated_image": None,
                    "detections": []
                }
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
        else:
            return {
                "success": False,
                "error": "Invalid input format. Expected file path or numpy array.",
                "original_image": None,
                "annotated_image": None,
                "detections": []
            }

        original_img = image.copy()
        annotated_img = image.copy()

        try:
            # Perform prediction
            results = self.model.predict(source=image, conf=conf_threshold, verbose=False)
            
            detections = []
            if results and len(results) > 0:
                result = results[0]

                # Generate Ultralytics annotated plot for visual comparison if needed
                # or draw custom clean bounding boxes
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for idx, box in enumerate(boxes):
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        class_name = self.class_names.get(cls_id, f"Class {cls_id}")

                        # Extract cropped plate
                        crop = crop_bbox(original_img, [x1, y1, x2, y2])

                        detection_info = {
                            "index": idx + 1,
                            "class_id": cls_id,
                            "class_name": class_name,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2],
                            "crop": crop
                        }
                        detections.append(detection_info)

                        # Draw bounding box and label on annotated_img
                        color = (0, 215, 255)  # Vibrant Amber / Cyan Accent (BGR)
                        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)

                        # Label header
                        label = f"{class_name}: {conf * 100:.1f}%"
                        (w_text, h_text), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        
                        # Background rectangle for text readability
                        label_y1 = max(0, y1 - h_text - 10)
                        cv2.rectangle(annotated_img, (x1, label_y1), (x1 + w_text + 10, label_y1 + h_text + 8), color, -1)
                        cv2.putText(annotated_img, label, (x1 + 5, label_y1 + h_text + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            return {
                "success": True,
                "error": "",
                "original_image": original_img,
                "annotated_image": annotated_img,
                "detections": detections
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Inference failed: {str(e)}",
                "original_image": original_img,
                "annotated_image": annotated_img,
                "detections": []
            }
