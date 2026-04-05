"""API v1 router configuration."""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    appointments,
    chat,
    dashboard,
    prediction,
    reviews,
    verification,
    yolo_with_gradcam,
)
from app.core.security import auth_backend
from app.core.users import fastapi_users
from app.schemas.user import UserCreate, UserRead, UserUpdate

api_v1_router = APIRouter(prefix="/api/v1")


# FastAPI-Users routers
api_v1_router.include_router(
    fastapi_users.get_auth_router(auth_backend, requires_verification=True),
    prefix="/auth",
    tags=["auth"],
)
api_v1_router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)
api_v1_router.include_router(
    fastapi_users.get_verify_router(UserRead),
    prefix="/auth",
    tags=["auth"],
)
api_v1_router.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate, requires_verification=True),
    prefix="/users",
    tags=["users"],
)
api_v1_router.include_router(
    fastapi_users.get_reset_password_router(),
    prefix="/auth",
    tags=["auth"],
)

# Custom routers
api_v1_router.include_router(
    prediction.router, prefix="/prediction", tags=["prediction"]
)
api_v1_router.include_router(
    yolo_with_gradcam.router, prefix="/yolo/detection", tags=["Yolo Gradcam"]
)
api_v1_router.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
api_v1_router.include_router(
    reviews.router, prefix="/prediction", tags=["prediction_reviews"]
)
api_v1_router.include_router(
    verification.router, prefix="/verification", tags=["verification"]
)
api_v1_router.include_router(
    appointments.router, prefix="/appointments", tags=["appointments"]
)
api_v1_router.include_router(chat.router, prefix="/chat", tags=["chat"])
