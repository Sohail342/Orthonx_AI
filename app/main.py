"""Main FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.api import api_v1_router
from app.core.config import settings


def create_application() -> FastAPI:
    """Create FastAPI app with middleware and routes."""

    application = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_DESCRIPTION,
        version="0.1.0",
        openapi_url=None
        if settings.ENVIRONMENT == "production"
        else f"{settings.API_V1_STR}/openapi.json",
        docs_url=None
        if settings.ENVIRONMENT == "production"
        else f"{settings.API_V1_STR}/docs",
        redoc_url=None
        if settings.ENVIRONMENT == "production"
        else f"{settings.API_V1_STR}/redoc",
    )

    # Set up CORS
    if settings.BACKEND_CORS_ORIGINS:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    application.include_router(api_v1_router)
    return application


app = create_application()


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "version": "0.1.0",
        "docs": f"{settings.API_V1_STR}/docs",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
