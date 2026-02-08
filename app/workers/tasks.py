"""Celery tasks for background email processing via SMTP."""

from app.core.config import settings
from app.utils.logging_utils import get_logger
from app.utils.mail_config import send_smtp_email
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(name="app.workers.tasks.send_verification_request", bind=True)
def send_verification_request(self, email: str, name: str, token: str) -> None:
    logger.info(f"Starting send_verification_request task for {email}")
    verify_url = (
        f"https://{settings.BACKEND_DOMAIN}/custom/auth/verify/email?token={token}"
        if settings.ENVIRONMENT == "production"
        else f"http://127.0.0.1:8000/custom/auth/verify/email?token={token}"
    )

    subject = "Verify your account"
    html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{
                    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                    background-color: #F5F7FF;
                    margin: 0;
                    padding: 0;
                    color: #15173D;
                }}
                .container {{
                    max-width: 600px;
                    margin: 40px auto;
                    background: #ffffff;
                    border-radius: 16px;
                    overflow: hidden;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
                    border-top: 6px solid #13ADC7;
                }}
                .content {{
                    padding: 40px 30px;
                    text-align: center;
                }}
                .brand-name {{
                    color: #13ADC7;
                    font-size: 28px;
                    font-weight: bold;
                    margin-bottom: 20px;
                    display: block;
                }}
                h2 {{
                    color: #15173D;
                    font-size: 24px;
                    margin-bottom: 15px;
                }}
                p {{
                    font-size: 16px;
                    line-height: 1.6;
                    color: #596080;
                    margin-bottom: 24px;
                }}
                /* Gradient Button Matching "Try Demo" Image */
                .btn {{
                    background: linear-gradient(90deg, #13ADC7 0%, #0098B0 100%);
                    color: #ffffff !important;
                    padding: 14px 35px;
                    border-radius: 50px;
                    text-decoration: none;
                    font-weight: 600;
                    font-size: 16px;
                    display: inline-block;
                    box-shadow: 0 4px 12px rgba(19, 173, 199, 0.3);
                }}
                .footer {{
                    background: #F9FAFB;
                    padding: 25px;
                    text-align: center;
                    border-top: 1px solid #E5E7EB;
                }}
                .footer p {{
                    font-size: 12px;
                    color: #9CA3AF;
                    margin: 5px 0;
                }}
                .link-text {{
                    color: #13ADC7;
                    word-break: break-all;
                    font-size: 13px;
                    text-decoration: none;
                }}
                @media only screen and (max-width: 480px) {{
                    .container {{ width: 95% !important; margin: 20px auto !important; }}
                    .content {{ padding: 30px 15px !important; }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="content">
                    <span class="brand-name">Orthonx</span>
                    <h2>Welcome to the future of clinical AI</h2>
                    <p>Hi <strong>{name}</strong>,</p>
                    <p>We're excited to help you streamline your workflow. Please verify your email to get full access to your clinical dashboard.</p>

                    <a href="{verify_url}" class="btn">Verify Account &nbsp; &rsaquo;</a>

                    <p style="margin-top: 35px; font-size: 13px; color: #9CA3AF;">
                        Or copy and paste this link into your browser:
                    </p>
                    <a href="{verify_url}" class="link-text">{verify_url}</a>
                </div>
                <div class="footer">
                    <p>&copy; {settings.PROJECT_NAME}. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """

    try:
        result = send_smtp_email(email, subject, html_content)
        logger.info(f"Verification email sent successfully to {email}: {result}")
        return result
    except Exception as e:
        logger.error(
            f"Failed to send verification email to {email}: {e}", exc_info=True
        )
        raise


@celery_app.task(name="app.workers.tasks.send_password_reset_email", bind=True)
def send_password_reset_email(self, email: str, name: str, token: str) -> None:
    logger.info(f"Starting send_password_reset_email task for {email}")
    reset_url = (
        f"https://{settings.BACKEND_DOMAIN}/custom/auth/reset-password?token={token}"
        if settings.ENVIRONMENT == "production"
        else f"http://127.0.0.1:8000/custom/auth/reset-password?token={token}"
    )

    subject = "Reset Your Password"
    html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{
                    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                    background-color: #F5F7FF;
                    margin: 0;
                    padding: 0;
                    color: #15173D;
                }}
                .container {{
                    max-width: 600px;
                    margin: 40px auto;
                    background: #ffffff;
                    border-radius: 16px;
                    overflow: hidden;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
                    border-top: 6px solid #13ADC7;
                }}
                .content {{
                    padding: 40px 30px;
                    text-align: center;
                }}
                .brand-name {{
                    color: #13ADC7;
                    font-size: 28px;
                    font-weight: bold;
                    margin-bottom: 25px;
                    display: block;
                }}
                h2 {{
                    color: #15173D;
                    font-size: 24px;
                    font-weight: 700;
                    margin-bottom: 20px;
                }}
                p {{
                    font-size: 16px;
                    line-height: 1.6;
                    color: #596080;
                    margin-bottom: 24px;
                }}
                /* Gradient Pill Button */
                .btn {{
                    background: linear-gradient(90deg, #13ADC7 0%, #0098B0 100%);
                    color: #ffffff !important;
                    padding: 14px 35px;
                    border-radius: 50px;
                    text-decoration: none;
                    font-weight: 600;
                    font-size: 16px;
                    display: inline-block;
                    box-shadow: 0 4px 12px rgba(19, 173, 199, 0.3);
                }}
                .footer {{
                    background: #F9FAFB;
                    padding: 25px;
                    text-align: center;
                    border-top: 1px solid #E5E7EB;
                }}
                .footer p {{
                    font-size: 12px;
                    color: #9CA3AF;
                    margin: 5px 0;
                }}
                .link-text {{
                    color: #13ADC7;
                    word-break: break-all;
                    font-size: 13px;
                    text-decoration: none;
                }}
                @media only screen and (max-width: 480px) {{
                    .container {{ width: 90% !important; margin: 20px auto !important; }}
                    .content {{ padding: 30px 20px !important; }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="content">
                    <span class="brand-name">Orthonx</span>
                    <h2>Reset your password</h2>
                    <p>Hello <strong>{name}</strong>,</p>
                    <p>We received a request to reset your password. No worries—it happens to the best of us! Click the button below to set a new one.</p>

                    <a href="{reset_url}" class="btn">Reset Password &nbsp; &rsaquo;</a>

                    <p style="margin-top: 35px; font-size: 13px; color: #9CA3AF;">
                        Button not working? Copy and paste this link:
                    </p>
                    <a href="{reset_url}" class="link-text">{reset_url}</a>

                    <p style="margin-top: 25px; font-size: 13px; color: #9CA3AF; font-style: italic;">
                        If you didn't request this, you can safely ignore this email. Your password will remain unchanged.
                    </p>
                </div>
                <div class="footer">
                    <p>&copy; {settings.PROJECT_NAME}. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """

    try:
        result = send_smtp_email(email, subject, html)
        logger.info(f"Password reset email sent successfully to {email}: {result}")
        return result
    except Exception as e:
        logger.error(
            f"Failed to send password reset email to {email}: {e}", exc_info=True
        )
        raise
