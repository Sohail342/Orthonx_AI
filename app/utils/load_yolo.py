import traceback

import numpy as np
import ultralytics
from ultralytics import YOLO

from app.utils.logging_utils import get_logger

MODEL = None

logger = get_logger(__name__)


def load_yolo_model() -> YOLO:
    """Load Yolo8 Model"""

    try:
        logger.info("Loading YOLOv8 model...")
        logger.info(f"Ultralytics version: {ultralytics.__version__}")
        MODEL = YOLO("app/ml_models/model.pt")
        logger.info("Model loaded successfully.")
        return MODEL
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(traceback.format_exc())
        raise e


def load_yolo_onnx() -> YOLO:
    """Load YOLO ONNX Model for CPU Inference"""
    global MODEL
    try:
        logger.info("Loading YOLO ONNX model...")
        logger.info(f"Ultralytics version: {ultralytics.__version__}")

        MODEL = YOLO("app/ml_models/final_bone_model.onnx", task="detect")

        # Warm-up
        logger.info("Warming up model...")
        dummy_img = np.zeros((1024, 1024, 3), dtype=np.uint8)
        MODEL(dummy_img, imgsz=1024, device="cpu")

        logger.info("Model loaded and warmed up successfully.")
        return MODEL
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(traceback.format_exc())
        raise e
