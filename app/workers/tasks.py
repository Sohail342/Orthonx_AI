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
        f"https://{settings.DOMAIN}/custom/auth/verify/email?token={token}"
        if settings.ENVIRONMENT == "production"
        else f"http://127.0.0.1:8000/custom/auth/verify/email?token={token}"
    )

    subject = "Verify your account"
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
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
                border-top: 5px solid #0062FF;
            }}
            .header {{
                background: #ffffff;
                padding: 40px 0 20px;
                text-align: center;
            }}
            .logo {{
                width: 180px;
                height: auto;
            }}
            .content {{
                padding: 20px 40px 40px;
                text-align: center;
            }}
            .content h2 {{
                color: #15173D;
                font-size: 24px;
                font-weight: 700;
                margin-bottom: 20px;
                letter-spacing: -0.5px;
            }}
            .content p {{
                font-size: 16px;
                line-height: 1.6;
                color: #596080;
                margin-bottom: 24px;
            }}
            .btn {{
                background: #0062FF;
                color: #ffffff;
                padding: 16px 32px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: 600;
                font-size: 16px;
                display: inline-block;
                margin: 10px 0;
                box-shadow: 0 4px 12px rgba(0, 98, 255, 0.2);
                transition: background-color 0.2s;
            }}
            .btn:hover {{
                background: #0048c2;
            }}
            .footer {{
                background: #F9FAFB;
                padding: 20px;
                text-align: center;
                border-top: 1px solid #E5E7EB;
            }}
            .footer p {{
                font-size: 12px;
                color: #9CA3AF;
                margin: 5px 0;
            }}
            .link-text {{
                color: #0062FF;
                word-break: break-all;
                font-size: 14px;
                text-decoration: none;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <img src="https://res.cloudinary.com/dms0a62ec/image/upload/v1766265717/pk06hclsmw465fdjcebl.jpg" alt="{settings.PROJECT_NAME}" class="logo">
            </div>
            <div class="content">
                <h2>Welcome to Orthonx!</h2>
                <p>Hi <strong>{name}</strong>,</p>
                <p>Thank you for signing up. Please verify your email address to access your dashboard and start using our clinical AI tools.</p>

                <a href="{verify_url}" class="btn">Verify Account</a>

                <p style="margin-top: 30px; font-size: 14px; color: #9CA3AF;">Or copy this link to your browser:</p>
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
        result = send_smtp_email(email, subject, html)
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
        f"https://{settings.DOMAIN}/custom/auth/reset-password?token={token}"
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
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
                border-top: 5px solid #0062FF;
            }}
            .header {{
                background: #ffffff;
                padding: 40px 0 20px;
                text-align: center;
            }}
            .logo {{
                width: 180px;
                height: auto;
            }}
            .content {{
                padding: 20px 40px 40px;
                text-align: center;
            }}
            .content h2 {{
                color: #15173D;
                font-size: 24px;
                font-weight: 700;
                margin-bottom: 20px;
                letter-spacing: -0.5px;
            }}
            .content p {{
                font-size: 16px;
                line-height: 1.6;
                color: #596080;
                margin-bottom: 24px;
            }}
            .btn {{
                background: #0062FF;
                color: #ffffff;
                padding: 16px 32px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: 600;
                font-size: 16px;
                display: inline-block;
                margin: 10px 0;
                box-shadow: 0 4px 12px rgba(0, 98, 255, 0.2);
                transition: background-color 0.2s;
            }}
            .btn:hover {{
                background: #0048c2;
            }}
            .footer {{
                background: #F9FAFB;
                padding: 20px;
                text-align: center;
                border-top: 1px solid #E5E7EB;
            }}
            .footer p {{
                font-size: 12px;
                color: #9CA3AF;
                margin: 5px 0;
            }}
            .link-text {{
                color: #0062FF;
                word-break: break-all;
                font-size: 14px;
                text-decoration: none;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <img src="https://res.cloudinary.com/dms0a62ec/image/upload/v1766265717/pk06hclsmw465fdjcebl.jpg" alt="{settings.PROJECT_NAME}" class="logo">
            </div>
            <div class="content">
                <h2>Reset Password</h2>
                <p>Hello <strong>{name}</strong>,</p>
                <p>We received a request to reset your password. Click the button below to choose a new one.</p>

                <a href="{reset_url}" class="btn">Reset Password</a>

                <p style="margin-top: 30px; font-size: 14px; color: #9CA3AF;">Or copy this link to your browser:</p>
                <a href="{reset_url}" class="link-text">{reset_url}</a>

                <p style="margin-top: 20px; font-size: 14px; color: #9CA3AF;">If you didn't request a password reset, you can safely ignore this email.</p>
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
