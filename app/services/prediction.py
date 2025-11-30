import base64
import io
import logging
from io import BytesIO

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from fastapi import File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from app.utils.gradcam import GradCAM

console_handler = logging.StreamHandler()

formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
console_handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)

BODY_PART_MODEL = "app/ml_models/bodypart_detector.onnx"
FRACTURE_MODEL = "app/ml_models/fracture_detector.onnx"
BODY_PARTS = ["ELBOW", "FINGER", "FOREARM", "HAND", "HUMERUS", "SHOULDER", "WRIST"]
MEAN = 0.2059
STD = 0.1768

# Initialize ONNX Runtime sessions
logger.info("Loading ONNX models...")
body_session = ort.InferenceSession(BODY_PART_MODEL, providers=["CPUExecutionProvider"])
frac_session = ort.InferenceSession(FRACTURE_MODEL, providers=["CPUExecutionProvider"])


class CNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.5) -> None:
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512 * 7 * 7, 512),  # 224 → 112 → 56 → 28 → 14 → 7
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        img = img.resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - MEAN) / STD
        arr = arr.reshape(1, 1, 224, 224)  # (N, C, H, W)
        return arr

    async def analyze_service(self, file: UploadFile = File(...)) -> JSONResponse:
        try:
            img_bytes = await file.read()
            x = self.preprocess_image(img_bytes)

            # Body Part Prediction
            body_logits = body_session.run(
                None, {body_session.get_inputs()[0].name: x}
            )[0]
            body_probs = np.exp(body_logits) / np.sum(np.exp(body_logits), axis=1)
            body_idx = int(np.argmax(body_logits, axis=1)[0])
            body_part = BODY_PARTS[body_idx]
            confidence = float(body_probs[0][body_idx])

            # Fracture Prediction
            frac_logit = float(
                frac_session.run(None, {frac_session.get_inputs()[0].name: x})[0][0][0]
            )
            frac_prob = float(1 / (1 + np.exp(-frac_logit)))

            response = {
                "body_part": body_part,
                "confidence": round(confidence, 4),
                "fracture": {
                    "probability": round(frac_prob, 4),
                    "positive": frac_prob > 0.5,
                    "risk": (
                        "HIGH"
                        if frac_prob > 0.7
                        else "MEDIUM"
                        if frac_prob > 0.5
                        else "LOW"
                    ),
                },
                "recommendation": (
                    "URGENT REVIEW"
                    if frac_prob > 0.7
                    else "Follow-up"
                    if frac_prob > 0.5
                    else "Likely normal"
                ),
            }

            return JSONResponse(response)

        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    async def analyze_with_gradcam_service(
        self, file: UploadFile = File(...)
    ) -> JSONResponse:
        try:
            img_bytes = await file.read()
            x_np = self.preprocess_image(img_bytes)  # (1,1,224,224) numpy
            x_torch = torch.from_numpy(x_np).float()

            # ONNX INFERENCE
            body_logits = body_session.run(
                None, {body_session.get_inputs()[0].name: x_np}
            )[0]
            body_probs = np.exp(body_logits) / np.sum(np.exp(body_logits), axis=1)
            body_idx = int(np.argmax(body_logits, axis=1)[0])
            body_part = BODY_PARTS[body_idx]

            frac_logit = float(
                frac_session.run(None, {frac_session.get_inputs()[0].name: x_np})[0][0][
                    0
                ]
            )
            frac_prob = 1 / (1 + np.exp(-frac_logit))
            is_positive = frac_prob > 0.5

            # ORIGINAL IMAGE (ALWAYS RETURNED)
            orig_img = Image.open(io.BytesIO(img_bytes)).convert("L")
            orig_img = orig_img.resize((224, 224))
            orig_np = np.array(orig_img)
            orig_np_rgb = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2RGB)

            # Encode original to base64
            orig_buf = BytesIO()
            Image.fromarray(orig_np_rgb).save(orig_buf, format="PNG")
            orig_b64 = base64.b64encode(orig_buf.getvalue()).decode("utf-8")

            # GRADCAM (ONLY IF POSITIVE)
            heatmap_b64 = None
            if is_positive:
                cam = gradcam.generate_cam(x_torch, target_class=0)
                cam = (cam * 255).astype(np.uint8)
                cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
                cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

                # Overlay with original
                overlay = cv2.addWeighted(orig_np_rgb, 0.6, cam, 0.4, 0)
                overlay_pil = Image.fromarray(overlay)

                # Encode heatmap to base64
                buf = BytesIO()
                overlay_pil.save(buf, format="PNG")
                heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            # RESPONSE
            response = {
                "body_part": body_part,
                "confidence": round(float(body_probs[0][body_idx]), 4),
                "fracture": {
                    "probability": round(frac_prob, 4),
                    "positive": bool(is_positive),
                    "risk": (
                        "HIGH"
                        if frac_prob > 0.7
                        else "MEDIUM"
                        if frac_prob > 0.5
                        else "LOW"
                    ),
                },
                "recommendation": (
                    "URGENT REVIEW"
                    if frac_prob > 0.7
                    else "Follow-up"
                    if frac_prob > 0.5
                    else "Likely normal"
                ),
                # Always return original
                "original_image": f"data:image/png;base64,{orig_b64}",
                # Only return heatmap if fracture is positive
                "heatmap": (
                    f"data:image/png;base64,{heatmap_b64}" if heatmap_b64 else None
                ),
            }

            return JSONResponse(response)

        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)


logger.info("Loading PyTorch fracture model for Grad-CAM...")
cnn_service = CNN(num_classes=1, dropout=0.6)
state_dict = torch.load("app/pytorch_models/fracture_model.pth", map_location="cpu")
state_dict = {
    k: v for k, v in state_dict.items() if not k.endswith(".num_batches_tracked")
}
cnn_service.load_state_dict(state_dict, strict=True)
cnn_service.eval()

# Initialize Grad-CAM on last ReLU in features
gradcam = GradCAM(cnn_service, target_layer_name="features.17")
