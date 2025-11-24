import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import cast

from app.core.config import settings

SMTP_HOST = settings.SMTP_HOST
SMTP_PORT = settings.SMTP_PORT
SMTP_USER = settings.SMTP_USER
SMTP_PASSWORD = settings.SMTP_PASSWORD
FROM_EMAIL = settings.EMAILS_FROM

logger = logging.getLogger(__name__)


def send_smtp_email(to_email: str, subject: str, html_body: str) -> dict:
    """Send an HTML email via SMTP (production-safe)."""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = FROM_EMAIL
        msg["To"] = to_email

        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(cast(str, SMTP_HOST), cast(int, SMTP_PORT)) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(cast(str, SMTP_USER), cast(str, SMTP_PASSWORD))
            server.send_message(msg)
            logger.info(f"Email sent to {to_email}")

        return {"status": "sent", "recipient": to_email}
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {e}")
        return {"status": "failed", "error": str(e)}
