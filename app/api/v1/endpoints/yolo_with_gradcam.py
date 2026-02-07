from uuid import uuid4

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.users import current_active_verified_user
from app.database.session import get_db
from app.models.users import User
from app.services.user_services import UserServices
from app.services.yolo_with_gradcam import yolo_grad_cam
from app.workers.celery_app import celery_app
from app.workers.yolo_tasks import detect_task

router = APIRouter()


@router.post("/detect")
async def detect(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(current_active_verified_user),
    file: UploadFile = File(...),
) -> dict:
    """Dectect Fracture with Yolo in background task"""
    image_bytes = await file.read()
    tast_result = detect_task.delay(image_bytes, user.id, f"detection_{uuid4()}")
    return {
        "message": "Detection task has been started in the background.",
        "task_id": tast_result.id,
    }


@router.get("/status/{task_id}")
async def task_status(
    task_id: str, user: User = Depends(current_active_verified_user)
) -> dict:
    """Check the status of a Celery task."""
    result = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "status": result.status,
    }

    if result.ready():
        response["result"] = result.result

    return response


@router.post("/detect/unauthenticated")
async def detect_unauthenticated(
    db: AsyncSession = Depends(get_db),
    file: UploadFile = File(...),
) -> dict:
    """Dectect Fracture with Yolo"""
    return await yolo_grad_cam.detect(db=db, file=file)


@router.get("/history")
async def get_detection_history(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(current_active_verified_user),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
) -> dict:
    """
    Get paginated detection history for the current user
    """
    user_services = UserServices(db=db)
    return await user_services.get_detection_history(
        user_id=user.id,
        page=page,
        page_size=page_size,
    )


@router.delete("/history/{record_id}")
async def delete_detection_record(
    record_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(current_active_verified_user),
) -> dict:
    """
    Delete a specific diagnosis record for the current user
    """
    user_services = UserServices(db=db)
    success = await user_services.delete_diagnosis_record(
        record_id=record_id, user_id=user.id
    )
    if not success:
        return {"message": "Failed to delete the record."}
    return {"message": "Record deleted successfully."}
