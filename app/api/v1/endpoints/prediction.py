from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import JSONResponse

from app.core.users import current_active_verified_user
from app.services.prediction import cnn_service

router = APIRouter(dependencies=[Depends(current_active_verified_user)])


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    try:
        return await cnn_service.analyze_service(file)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/analyze_with_gradcam")
async def analyze_with_gradcam(file: UploadFile = File(...)) -> JSONResponse:
    try:
        return await cnn_service.analyze_with_gradcam_service(file)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
