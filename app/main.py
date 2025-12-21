"""Main FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.api import api_v1_router
from app.api.v1.endpoints.custom_auth import verify_router
from app.core.config import settings

CORS = [str(origin)[:-1] for origin in settings.BACKEND_CORS_ORIGINS]


def create_application() -> FastAPI:
    """Create FastAPI app with middleware and routes."""

    application = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_DESCRIPTION,
        version="0.1.0",
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url=f"{settings.API_V1_STR}/docs",
        redoc_url=f"{settings.API_V1_STR}/redoc",
    )

    # Set up CORS
    if settings.BACKEND_CORS_ORIGINS:
        if settings.ENVIRONMENT == "development":
            application.add_middleware(
                CORSMiddleware,
                allow_origins=CORS,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        else:
            application.add_middleware(
                CORSMiddleware,
                allow_origins=CORS,
                allow_credentials=True,
                allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
                allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
                expose_headers=["X-Process-Time"],
                max_age=3600,
            )
    application.include_router(api_v1_router)
    application.include_router(
        verify_router, prefix="/custom", tags=["Custom Auth for Verification"]
    )
    return application


app = create_application()


@app.get("/")
def root() -> dict:
    """Root endpoint."""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "version": "0.1.0",
        "docs": f"{settings.API_V1_STR}/docs",
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
