import traceback
from typing import Optional

from app.services.new_yolo_service import BoneFractureDetector
from app.utils.logging_utils import get_logger

MODEL: Optional[BoneFractureDetector] = None

logger = get_logger(__name__)


def load_yolo_model() -> BoneFractureDetector:
    """Load Yolo BoneFractureDetector"""
    global MODEL
    if MODEL is not None:
        return MODEL

    try:
        logger.info("Initializing BoneFractureDetector with YOLOv8 model...")
        MODEL = BoneFractureDetector(
            model_path="app/ml_models/best.pt",
            conf_threshold=0.25,
            device="cpu",  # Adjust as needed (e.g., "cuda" if available)
        )
        logger.info("Model loaded successfully.")
        return MODEL
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(traceback.format_exc())
        raise e
