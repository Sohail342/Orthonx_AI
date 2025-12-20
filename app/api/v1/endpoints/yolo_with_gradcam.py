from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse

from app.services.yolo_with_gradcam import yolo_grad_cam

router = APIRouter()


@router.post("/detect")
async def detect(file: UploadFile = File(...)) -> dict:
    """Dectect Fracture with Yolo"""
    return await yolo_grad_cam.detect(file=file)


@router.get("/gradcam/{image_id}")
async def get_gradcam(image_id: str) -> FileResponse:
    """
    Generate custom Grad-CAM visualization for a previously detected image
    """
    return await yolo_grad_cam.get_gradcam(image_id=image_id)
