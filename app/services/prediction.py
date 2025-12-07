import base64
import io

import numpy as np
import torch
from fastapi import File, HTTPException, UploadFile
from PIL import Image, ImageDraw

from app.utils.load_models import (
    CLASS_ID_TO_LABEL,
    class_processor,
    ort_class_model,
    ort_session,
)
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BoneFracturePrediction:
    def __init__(self) -> None:
        self.MODEL_SIZE = 800  # DETR uses 800 pixels for the longest side
        self.MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(
        self, image: Image.Image
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        w, h = image.size
        scale = self.MODEL_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image_resized = image.resize((new_w, new_h))
        img_array = np.array(image_resized).astype(np.float32) / 255.0
        img_array = (img_array - self.MEAN) / self.STD
        padded = np.zeros((self.MODEL_SIZE, self.MODEL_SIZE, 3), dtype=np.float32)
        padded[:new_h, :new_w, :] = img_array
        pixel_values = np.transpose(padded, (2, 0, 1))
        return np.expand_dims(pixel_values, axis=0), (w, h), (new_w, new_h)

    def postprocess(
        self,
        logits: np.ndarray,
        boxes: np.ndarray,
        original_size: tuple[int, int],
        new_size: tuple[int, int],
        threshold: float = 0.05,
    ) -> list[list[float]]:
        orig_w, orig_h = original_size
        new_w, new_h = new_size

        # Calculate the single scale factor for denormalization
        scale_factor = max(orig_h, orig_w) / self.MODEL_SIZE

        logits_tensor = torch.tensor(logits)
        boxes_tensor = torch.tensor(boxes)

        probs = logits_tensor.softmax(-1)[..., 1]
        keep = probs > threshold
        kept_boxes = boxes_tensor[keep].numpy()

        fracture_boxes = []

        for box in kept_boxes:
            # 1. Convert from normalized [x_c, y_c, w, h] to [x0, y0, x1, y1]
            x_c, y_c, w, h = box

            x0_norm = x_c - w / 2
            y0_norm = y_c - h / 2
            x1_norm = x_c + w / 2
            y1_norm = y_c + h / 2

            # Convert normalized [0, 1] coordinates to 800x800 pixel space:
            x0_800 = x0_norm * self.MODEL_SIZE
            x1_800 = x1_norm * self.MODEL_SIZE
            y0_800 = y0_norm * self.MODEL_SIZE
            y1_800 = y1_norm * self.MODEL_SIZE

            # map from 800x800 space back to original space using the inverse scale factor
            final_x0 = x0_800 * scale_factor
            final_x1 = x1_800 * scale_factor
            final_y0 = y0_800 * scale_factor
            final_y1 = y1_800 * scale_factor

            # Final checks and type casting:
            final_x0 = max(0.0, min(final_x0, final_x1))
            final_y0 = max(0.0, min(final_y0, final_y1))
            final_x1 = min(float(orig_w), max(final_x0, final_x1))
            final_y1 = min(float(orig_h), max(final_y0, final_y1))

            fracture_boxes.append(
                [float(final_x0), float(final_y0), float(final_x1), float(final_y1)]
            )

        return fracture_boxes

    async def run_inference_detection(self, image_file: Image.Image) -> dict:
        try:
            contents = await image_file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")

            original_buffer = io.BytesIO()
            image.save(original_buffer, format="PNG")
            original_img_str = base64.b64encode(original_buffer.getvalue()).decode(
                "utf-8"
            )

            # Preprocess
            pixel_values, orig_size, scale = self.preprocess(image)

            # ONNX inference
            outputs = ort_session.run(None, {"pixel_values": pixel_values})
            logits = outputs[0]
            boxes = outputs[1]

            # Post-process
            fracture_boxes = self.postprocess(logits, boxes, orig_size, scale)

            # Draw boxes on image
            annotated_image = image.copy()
            draw = ImageDraw.Draw(annotated_image)
            for box in fracture_boxes:
                draw.rectangle(box, outline="red", width=3)

            annotated_buffer = io.BytesIO()
            annotated_image.save(annotated_buffer, format="PNG")
            annotated_img_str = base64.b64encode(annotated_buffer.getvalue()).decode(
                "utf-8"
            )

            return {
                "original_image": original_img_str,
                "annotated_image": annotated_img_str,
                "fracture_boxes": fracture_boxes,
            }
        except Exception as e:
            raise HTTPException(500, f"Internal Server Error (Detection): {e}")

    async def run_inference_fracture(self, file: UploadFile = File(...)) -> dict:
        if not file.content_type.startswith("image/"):
            raise HTTPException(400, "File must be an image.")

        try:
            img_bytes = await file.read()
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            inputs = class_processor(images=image, return_tensors="np")
            ort_inputs = {"pixel_values": inputs["pixel_values"]}

            outputs = ort_class_model(**ort_inputs)
            logits = outputs["logits"]

            probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
            cls_idx = int(np.argmax(logits, axis=1)[0])
            confidence = float(probs[0][cls_idx] * 100)

            return {
                "filename": file.filename,
                "prediction": CLASS_ID_TO_LABEL[cls_idx],
                "confidence_percent": f"{confidence:.2f}%",
            }

        except Exception as e:
            raise HTTPException(500, f"Internal Server Error (Classification): {e}")


bone_fracture_predictor = BoneFracturePrediction()
