from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from app.ml_models.load_model import model_load
from app.services.predict_fracture import predict_fracture_service

router = APIRouter()


@router.post("/predict/")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """Endpoint to predict fracture from uploaded image file."""

    try:
        model_path = "app/ml_models/fracture_body_best.keras"
        model = model_load(model_path)
        result = await predict_fracture_service(model, file)
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
