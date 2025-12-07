from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from app.services.prediction import bone_fracture_predictor

router = APIRouter()


@router.post("/detect")
async def detect(image_file: UploadFile = File(...)) -> JSONResponse:
    result = await bone_fracture_predictor.run_inference_detection(image_file)
    return JSONResponse(result)


@router.post("/predict")
async def predict_fracture(image_file: UploadFile = File(...)) -> JSONResponse:
    result = await bone_fracture_predictor.run_inference_fracture(image_file)
    return JSONResponse(result)
