import io
from typing import Any

import numpy as np
from fastapi import UploadFile
from PIL import Image

from app.ml_models.load_model import BODY_PARTS

IMG_SIZE = (224, 224)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess raw image bytes into a normalized NumPy array ready for model input."""

    # Open and force RGB
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize to training resolution
    img = img.resize((224, 224))

    # Convert to float32 array normalized
    img = np.asarray(img, dtype=np.float32) / 255.0

    # Ensure shape is (224, 224, 3)
    if img.ndim == 2:
        img = np.stack((img,) * 3, axis=-1)
    elif img.shape[-1] != 3:
        img = img[..., :3]

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


async def predict_fracture_service(model: Any, file: UploadFile) -> dict:
    """Predict fracture from uploaded image file."""
    try:
        image_bytes = await file.read()
        img = preprocess_image(image_bytes)

        body_pred, abnormal_pred = model.predict(img)

        body_idx = int(np.argmax(body_pred, axis=1)[0])
        body_part = BODY_PARTS[body_idx]

        abnormal_score = float(abnormal_pred[0][0])
        abnormal_label = "Abnormal" if abnormal_score > 0.5 else "Normal"

        return {
            "body_part": body_part,
            "abnormality": abnormal_label,
            "abnormality_score": round(abnormal_score, 4),
        }
    except Exception as e:
        return {"error": str(e)}
