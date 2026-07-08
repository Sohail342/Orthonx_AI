"""Main FastAPI application."""

import os
import sys
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.api import api_v1_router
from app.api.v1.endpoints.custom_auth import verify_router
from app.core.config import settings
from app.utils.logging_utils import get_logger
from app.utils.ollama_utils import OLLAMA_BASE_URL, OLLAMA_MODEL

CORS = [str(origin)[:-1] for origin in settings.BACKEND_CORS_ORIGINS]


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Lifespan handler to initialize/warmup services."""
    logger.info("Initializing services on startup...")

    if "pytest" in sys.modules:
        logger.info("Test environment detected. Skipping AI Model warmup.")
        yield
        logger.info("Shutting down services...")
        return

    # Read Knowledge Base for Prompt Priming
    knowledge_path = os.path.join(
        os.path.dirname(__file__), "static", "CHATBOT_KNOWLEDGE.md"
    )
    kb_content = ""
    if os.path.exists(knowledge_path):
        try:
            with open(knowledge_path, "r", encoding="utf-8") as f:
                kb_content = f.read()
        except Exception as e:
            logger.error(f"Failed to read knowledge base for warmup: {e}")

    # Warm up Ollama model (Aggressive & Primed)
    try:
        logger.info(f"Warming up Ollama model: {OLLAMA_MODEL} (max 3min)")

        warmup_prompt = (
            "You are an AI assistant for the Orthonx platform. Your name is 'Orthonx Assistant'. "
            "Use the knowledge base below for future answers.\n\n"
            f"### KNOWLEDGE BASE CONTENT ###\n{kb_content}\n\n"
            "SYSTEM: Init complete. Respond only with 'READY'."
        )

        async with httpx.AsyncClient(timeout=180.0) as client:
            await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": warmup_prompt,
                    "stream": False,
                    "keep_alive": -1,  # Keep in memory indefinitely
                },
            )
        logger.info("AI Model primed and persistent in memory.")
    except Exception as e:
        logger.warning(f"AI Model warmup/priming failed: {e}")

    yield

    logger.info("Shutting down services...")


def create_application() -> FastAPI:
    """Create FastAPI app with middleware and routes."""

    application = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_DESCRIPTION,
        version="0.1.0",
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url=f"{settings.API_V1_STR}/docs",
        redoc_url=f"{settings.API_V1_STR}/redoc",
        lifespan=lifespan,
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
