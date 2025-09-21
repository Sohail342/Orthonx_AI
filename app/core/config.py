"""Application configuration."""

import secrets
from typing import Any, List, Optional, Union

from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    ALGORITHM: str = "HS256"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    DOMAIN: str = "domain.com"
    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    PROJECT_NAME: str = "fyp_backend"
    PROJECT_YEAR: str = "2025"
    PROJECT_DESCRIPTION: str = "FYP"

    DATABASE_URL: Optional[str] = None
    SYNC_DATABASE_URL: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        """Set default DB URLs if missing."""
        if not self.DATABASE_URL:
            self.DATABASE_URL = (
                "sqlite+aiosqlite:///./app.db"
                if self.ENVIRONMENT == "development"
                else "postgresql+asyncpg://postgres:postgres@db:5432/fyp_backend"
            )
        if not self.SYNC_DATABASE_URL:
            self.SYNC_DATABASE_URL = (
                "sqlite:///./app.db"
                if self.ENVIRONMENT == "development"
                else "postgresql://postgres:postgres@db:5432/fyp_backend"
            )

    SMTP_PORT: Optional[int] = None
    SMTP_HOST: Optional[str] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAILS_FROM: Optional[str] = "fyp_backend@gmail.com"

    # Redis / Celery
    REDIS_URL: str = "redis://redis:6379"
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore",
    }


settings = Settings()
