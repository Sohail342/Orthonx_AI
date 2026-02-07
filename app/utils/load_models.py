import os
from pathlib import Path

import onnxruntime as ort
from huggingface_hub import hf_hub_download, snapshot_download
from optimum.onnxruntime import ORTModelForImageClassification
from transformers import AutoImageProcessor

from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

HF_TOKEN = os.getenv("HF_API_KEY")

detection_dir = Path("app/ml_models/bone_fracture_detection")
detection_model_path = detection_dir / "bone_fracture_detection.onnx"

try:
    if not detection_model_path.exists():
        logger.info(f"Downloading detection model to {detection_dir}...")

        # Download the .onnx file
        hf_hub_download(
            repo_id="sohail342/bone_fracture_detection",
            filename="bone_fracture_detection.onnx",
            local_dir=str(detection_dir),
            local_dir_use_symlinks=False,
            token=HF_TOKEN,
        )

        hf_hub_download(
            repo_id="sohail342/bone_fracture_detection",
            filename="bone_fracture_detection.onnx.data",
            local_dir=str(detection_dir),
            local_dir_use_symlinks=False,
            token=HF_TOKEN,
        )

    ort_session = ort.InferenceSession(str(detection_model_path))
    logger.info("ONNX detection model loaded successfully.")

except Exception as e:
    logger.error(f"Error loading ONNX model: {e}")
    raise e


class_model_dir = Path("app/ml_models/bone_fracture_model_onnx")

try:
    if not class_model_dir.exists() or not any(class_model_dir.iterdir()):
        logger.info(
            f"Local classification model not found at {class_model_dir}. Downloading..."
        )

        snapshot_download(
            repo_id="sohail342/bone_fracture_model_onnx",
            local_dir=str(class_model_dir),
            local_dir_use_symlinks=False,
            token=HF_TOKEN,
        )
    else:
        logger.info(f"Local classification model found at {class_model_dir}.")

    # Load from the directory path
    ort_class_model = ORTModelForImageClassification.from_pretrained(
        str(class_model_dir)
    )
    class_processor = AutoImageProcessor.from_pretrained(str(class_model_dir))
    CLASS_ID_TO_LABEL = ort_class_model.config.id2label

    logger.info("Successfully loaded Classification Model")

except Exception as e:
    logger.error(f"Error loading Classification Model: {e}")
    raise e
