import io

import matplotlib.pyplot as plt
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from app.services.prediction import FractureDetector

detector = FractureDetector("app/ml_models/fracture_detection_model.onnx")

# FastAPI App
router = APIRouter()


@router.post("/predict")
async def predict_fracture(file: UploadFile = File(...)) -> JSONResponse:
    """Predict fracture with region detection"""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Predict
        result = detector.predict(image)

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict_with_visualization")
async def predict_with_visualization(file: UploadFile = File(...)) -> JSONResponse:
    """Predict with visualization data"""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Predict
        result = detector.predict(image)

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Original image
        ax1.imshow(image)
        ax1.set_title("Original Image")
        ax1.axis("off")

        # Image with bounding boxes
        ax2.imshow(image)
        for region in result["fracture_regions"]:
            x, y, w, h = region["bbox"]
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2)
            ax2.add_patch(rect)
            ax2.text(
                x, y - 5, f"Conf: {region['confidence']:.2f}", color="red", fontsize=8
            )

        ax2.set_title(f"Detected Regions: {result['regions_count']}")
        ax2.axis("off")

        # Save visualization to bytes
        img_buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img_buf, format="png", dpi=100, bbox_inches="tight")
        img_buf.seek(0)
        plt.close()

        # Convert to base64 for API response
        import base64

        visualization_base64 = base64.b64encode(img_buf.getvalue()).decode()

        result["visualization"] = f"data:image/png;base64,{visualization_base64}"

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
