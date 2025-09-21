from fastapi_mail import ConnectionConfig

from app.core.config import settings

conf = ConnectionConfig(
    MAIL_USERNAME=settings.SMTP_USER,
    MAIL_PASSWORD=settings.SMTP_PASSWORD,
    MAIL_FROM=settings.EMAILS_FROM,
    MAIL_PORT=587,
    MAIL_SERVER=settings.SMTP_HOST,
    MAIL_FROM_NAME="FYP",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True,
)
