import traceback

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
