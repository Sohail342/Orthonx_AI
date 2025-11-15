import base64
import io
from io import BytesIO

import cv2
import numpy as np
import onnxruntime as ort
import torch
from fastapi import File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from app.utils.gradcam import GradCAM

from .cnn import ImprovedCNN

BODY_PART_MODEL = "app/ml_models/bodypart_detector.onnx"
FRACTURE_MODEL = "app/ml_models/fracture_detector.onnx"
BODY_PARTS = ["ELBOW", "FINGER", "FOREARM", "HAND", "HUMERUS", "SHOULDER", "WRIST"]
MEAN = 0.2059
STD = 0.1768

print("Loading ONNX models...")
body_session = ort.InferenceSession(BODY_PART_MODEL, providers=["CPUExecutionProvider"])
frac_session = ort.InferenceSession(FRACTURE_MODEL, providers=["CPUExecutionProvider"])

# === GLOBAL: PYTORCH + GRAD-CAM ===
print("Loading PyTorch fracture model for Grad-CAM...")
frac_pth_model = ImprovedCNN(num_classes=1, dropout=0.6)
state_dict = torch.load("app/pytorch_models/fracture_model.pth", map_location="cpu")
state_dict = {
    k: v for k, v in state_dict.items() if not k.endswith(".num_batches_tracked")
}
frac_pth_model.load_state_dict(state_dict, strict=True)
frac_pth_model.eval()

# Initialize Grad-CAM on last ReLU in features
gradcam = GradCAM(frac_pth_model, target_layer_name="features.17")
print("Grad-CAM ready on features.17")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.reshape(1, 1, 224, 224)  # (N, C, H, W)
    return arr


async def analyze_service(file: UploadFile = File(...)) -> JSONResponse:
    try:
        img_bytes = await file.read()
        x = preprocess_image(img_bytes)

        # Body Part Prediction
        body_logits = body_session.run(None, {body_session.get_inputs()[0].name: x})[0]
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


async def analyze_with_gradcam_service(file: UploadFile = File(...)) -> JSONResponse:
    try:
        img_bytes = await file.read()
        x_np = preprocess_image(img_bytes)  # (1,1,224,224) numpy
        x_torch = torch.from_numpy(x_np).float()

        # === ONNX INFERENCE (fast) ===
        body_logits = body_session.run(None, {body_session.get_inputs()[0].name: x_np})[
            0
        ]
        body_probs = np.exp(body_logits) / np.sum(np.exp(body_logits), axis=1)
        body_idx = int(np.argmax(body_logits, axis=1)[0])
        body_part = BODY_PARTS[body_idx]

        frac_logit = float(
            frac_session.run(None, {frac_session.get_inputs()[0].name: x_np})[0][0][0]
        )
        frac_prob = 1 / (1 + np.exp(-frac_logit))
        is_positive = frac_prob > 0.5

        heatmap_b64 = None
        if is_positive:
            cam = gradcam.generate_cam(x_torch, target_class=0)
            cam = (cam * 255).astype(np.uint8)
            cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
            cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

            # Load original image
            orig_img = Image.open(io.BytesIO(img_bytes)).convert("L")
            orig_img = orig_img.resize((224, 224))
            orig_np = np.array(orig_img)
            orig_np = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2RGB)

            # Overlay
            overlay = cv2.addWeighted(orig_np, 0.6, cam, 0.4, 0)
            overlay_pil = Image.fromarray(overlay)

            # Encode to base64
            buf = BytesIO()
            overlay_pil.save(buf, format="PNG")
            heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        response = {
            "body_part": body_part,
            "confidence": round(float(body_probs[0][body_idx]), 4),
            "fracture": {
                "probability": round(frac_prob, 4),
                "positive": bool(frac_prob > 0.5),
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
            "heatmap": f"data:image/png;base64,{heatmap_b64}" if heatmap_b64 else None,
        }

        return JSONResponse(response)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
