"""
Production-Ready Bone Fracture Detection System
Complete inference pipeline with visualization and reporting
"""

import io
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import requests
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BoneFractureDetector:
    """
    Production-ready bone fracture detection system using YOLOv8
    """

    # Class mapping
    CLASS_NAMES = {
        0: "ELBOW",
        1: "FINGER",
        2: "FOREARM",
        3: "HAND",
        4: "HUMERUS",
        5: "SHOULDER",
        6: "WRIST",
    }

    # Color map for visualization
    COLORS = {
        0: (255, 0, 0),  # ELBOW - Red
        1: (0, 255, 0),  # FINGER - Green
        2: (0, 0, 255),  # FOREARM - Blue
        3: (255, 255, 0),  # HAND - Yellow
        4: (255, 0, 255),  # HUMERUS - Magenta
        5: (0, 255, 255),  # SHOULDER - Cyan
        6: (255, 128, 0),  # WRIST - Orange
    }

    def __init__(
        self,
        model_path: str = "app/ml_models/best.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        """
        Initialize the detector

        Args:
            model_path: Path to trained YOLO model
            conf_threshold: Confidence threshold (0.0-1.0)
            iou_threshold: IoU threshold for NMS
            device: 'cuda' or 'cpu'
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        logger.info(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)
        logger.info(f"Model loaded successfully on {device}")

    def predict_url(
        self, image_url: str, save: bool = False, show: bool = False
    ) -> Dict:
        """
        Predict fractures from image URL

        Args:
            image_url: URL of X-ray image
            save: Save annotated image
            show: Display result

        Returns:
            Dictionary with detection results
        """
        logger.info(f"Downloading image from URL: {image_url}")

        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image_bytes = response.content

            # Run prediction
            result = self.predict_bytes(image_bytes, save=save, show=show)
            result["source"] = image_url

            return result

        except Exception as e:
            logger.error(f"Error downloading image: {e}")
            return {"error": str(e)}

    def predict_bytes(
        self, image_bytes: bytes, save: bool = False, show: bool = False
    ) -> Dict:
        """
        Predict fractures from image bytes

        Args:
            image_bytes: Bytes of X-ray image
            save: Save annotated image
            show: Display result

        Returns:
            Dictionary with detection results
        """
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Save temporarily for YOLO if needed, but YOLO can take PIL or numpy
        results = self.model.predict(
            source=img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        # Parse results
        detections = self._parse_results(results[0])

        # Create annotated image
        annotated_img = self._create_annotated_image_from_pil(img, detections)

        if show:
            annotated_img.show()

        if save:
            save_path = self._save_results(annotated_img, detections)
            logger.info(f"Results saved to: {save_path}")

        return {
            "num_detections": len(detections),
            "detections": detections,
            "annotated_image": annotated_img,
            "timestamp": datetime.now().isoformat(),
        }

    def predict_image(
        self, image_path: str, save: bool = False, show: bool = False
    ) -> Dict:
        """
        Predict fractures from local image

        Args:
            image_path: Path to X-ray image
            save: Save annotated image
            show: Display result

        Returns:
            Dictionary with detection results
        """
        logger.info(f"Running fracture detection on: {image_path}")

        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        # Parse results
        detections = self._parse_results(results[0])

        # Create annotated image
        annotated_img = self._create_annotated_image(image_path, detections)

        if show:
            annotated_img.show()

        if save:
            save_path = self._save_results(annotated_img, detections)
            logger.info(f"Results saved to: {save_path}")

        return {
            "image": image_path,
            "num_detections": len(detections),
            "detections": detections,
            "annotated_image": annotated_img,
            "timestamp": datetime.now().isoformat(),
        }

    def _parse_results(self, result) -> List[Dict]:
        """Parse YOLO results into structured format"""
        detections = []

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detection = {
                "class_id": int(box.cls[0]),
                "class": self.CLASS_NAMES.get(int(box.cls[0]), "Unknown"),
                "confidence": round(float(box.conf[0]), 2),
                "box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                "bbox": [x1, y1, x2, y2],  # Keep for backward compatibility if needed
                "bbox_normalized": box.xywhn[0].tolist(),
            }
            detections.append(detection)

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)

        return detections

    def _create_annotated_image(
        self, image_path: str, detections: List[Dict]
    ) -> Image.Image:
        """Create annotated image with bounding boxes and labels"""
        img = Image.open(image_path).convert("RGB")
        return self._create_annotated_image_from_pil(img, detections)

    def _create_annotated_image_from_pil(
        self, img: Image.Image, detections: List[Dict]
    ) -> Image.Image:
        """Create annotated image from PIL Image"""
        draw = ImageDraw.Draw(img)

        # Try to load a font
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
            )
        except Exception as e:
            logger.error(f"Error occurred while loading font: {e}")
            font = ImageFont.load_default()

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            class_id = det["class_id"]
            conf = det["confidence"]
            class_name = det["class"]

            # Get color
            color = self.COLORS.get(class_id, (255, 255, 255))

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Draw label background
            label = f"{class_name}: {conf:.1%}"
            try:
                bbox = draw.textbbox((x1, y1), label, font=font)
                draw.rectangle(bbox, fill=color)
                draw.text((x1, y1), label, fill=(0, 0, 0), font=font)
            except Exception as e:
                logger.error(f"Error occurred while drawing text: {e}")
                draw.text((x1, y1), label, fill=color)

        return img

    def _save_results(self, annotated_img: Image.Image, detections: List[Dict]) -> str:
        """Save annotated image and detection JSON"""
        output_dir = Path("detection_results")
        output_dir.mkdir(exist_ok=True)

        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = output_dir / f"detection_{timestamp}.jpg"
        annotated_img.save(img_path, quality=95)

        # Save JSON
        json_path = output_dir / f"detection_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(detections, f, indent=2)

        return str(output_dir)

    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "classes": self.CLASS_NAMES,
        }
