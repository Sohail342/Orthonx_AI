from pathlib import Path

import onnxruntime as ort
from huggingface_hub import hf_hub_download, snapshot_download
from optimum.onnxruntime import ORTModelForImageClassification
from transformers import AutoImageProcessor

from app.core.config import settings
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


HF_TOKEN = settings.HF_API_KEY

try:
    model_path = Path(
        "app/ml_models/bone_fracture_detection/bone_fracture_detection.onnx"
    )
    if not model_path.exists():
        # Download Bone Fracture detection ONNX model
        detection_model_path = hf_hub_download(
            repo_id="sohail342/bone_fracture_detection",
            filename="bone_fracture_detection.onnx",
            cache_dir="app/ml_models/bone_fracture_detection",
            token=HF_TOKEN,
        )

        # Download Load data file
        hf_hub_download(
            repo_id="sohail342/bone_fracture_detection",
            filename="bone_fracture_detection.onnx.data",
            cache_dir="app/ml_models/bone_fracture_detection",
            token=HF_TOKEN,
        )
    else:
        detection_model_path = str(model_path)

    # Load ONNX model
    ort_session = ort.InferenceSession(detection_model_path)
    logger.info("ONNX model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading ONNX model: {e}")
    raise e


# Classification Model Setup
model_dir_path = Path("app/ml_models/bone_fracture_model_onnx")

# Use a variable for the path that contains ALL the files
CLASS_ONNX_PATH = str(model_dir_path)

try:
    if not model_dir_path.exists() or not any(model_dir_path.iterdir()):
        logger.info(f"Local model not found at {model_dir_path}. Downloading...")

        # Use snapshot_download to download ALL files to the directory
        snapshot_download(
            repo_id="sohail342/bone_fracture_model_onnx",
            local_dir=str(model_dir_path),
            local_dir_use_symlinks=False,
            token=HF_TOKEN,
        )

    else:
        logger.info(f"Local model found at {model_dir_path}.")

    DETR_CONFIDENCE_THRESHOLD = 0.85

    # These functions are designed to accept a directory path (which CLASS_ONNX_PATH now is)
    ort_class_model = ORTModelForImageClassification.from_pretrained(CLASS_ONNX_PATH)
    class_processor = AutoImageProcessor.from_pretrained(CLASS_ONNX_PATH)
    CLASS_ID_TO_LABEL = ort_class_model.config.id2label
    logger.info("Successfully loaded Classification Model")

except Exception as e:
    logger.error(f"Error loading Classification Model: {e}")
