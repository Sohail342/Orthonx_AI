"""Email configuration for sending emails."""

from typing import Dict

import resend

from app.core.config import settings


class EmailConfig:
    def __init__(
        self,
        resend_api_key: str,
        from_email: str = f"Orthonx <support@{settings.DOMAIN}>",
    ):
        resend.api_key = resend_api_key
        self.FROM_EMAIL: str = from_email

    def send_mail(self, html_content: str, to_email: str, subject: str) -> Dict:
        params: resend.Emails.SendParams = {
            "from": self.FROM_EMAIL,
            "to": [to_email],
            "subject": subject,
            "html": html_content,
        }
        email: resend.Emails.SendResponse = resend.Emails.send(params)
        return email
