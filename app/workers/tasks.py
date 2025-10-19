"""Celery tasks for background email processing via SMTP."""

from celery import shared_task

from app.core.config import settings
from app.utils.mail_config import send_smtp_email


@shared_task
def send_verification_request(email: str, name: str, token: str) -> None:
    verify_url = (
        f"https://{settings.DOMAIN}/custom/auth/verify/email?token={token}"
        if settings.ENVIRONMENT == "production"
        else f"http://127.0.0.1:8000/custom/auth/verify/email?token={token}"
    )

    subject = "Verify your account"
    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; background-color:#f9fafb; color:#111827;">
        <table style="max-width:600px;margin:auto;background:#fff;border-radius:8px;padding:24px;">
            <tr><td style="text-align:center;">
                <h1 style="color:#4f46e5;">Welcome to {settings.PROJECT_NAME} 🎉</h1>
                <p>Hi <strong>{name}</strong>,</p>
                <p>Thanks for signing up! Please verify your account by clicking below:</p>
                <a href="{verify_url}" style="background:#4f46e5;color:#fff;padding:12px 24px;border-radius:8px;text-decoration:none;font-weight:bold;">
                    Verify Account
                </a>
                <p style="font-size:12px;color:#6b7280;">If the button doesn't work, copy this link:<br>
                <a href="{verify_url}" style="color:#4f46e5;">{verify_url}</a></p>
            </td></tr>
        </table>
    </body>
    </html>
    """

    send_smtp_email(email, subject, html)


@shared_task
def send_password_reset_email(email: str, name: str, token: str) -> None:
    reset_url = (
        f"https://{settings.DOMAIN}/custom/auth/reset-password?token={token}"
        if settings.ENVIRONMENT == "production"
        else f"http://127.0.0.1:8000/custom/auth/reset-password?token={token}"
    )

    subject = "Reset Your Password"
    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; background-color:#f9fafb; color:#111827;">
        <table style="max-width:600px;margin:auto;background:#fff;border-radius:8px;padding:24px;">
            <tr><td style="text-align:center;">
                <h1 style="color:#4f46e5;">Reset Your Password</h1>
                <p>Hello <strong>{name}</strong>,</p>
                <p>Click the button below to reset your password:</p>
                <a href="{reset_url}" style="background:#4f46e5;color:#fff;padding:12px 24px;border-radius:8px;text-decoration:none;font-weight:bold;">
                    Reset Password
                </a>
                <p style="font-size:12px;color:#6b7280;">If you didn’t request this, ignore this email.</p>
                <p style="font-size:12px;color:#6b7280;">Or copy this link:<br>
                <a href="{reset_url}" style="color:#4f46e5;">{reset_url}</a></p>
            </td></tr>
        </table>
    </body>
    </html>
    """

    send_smtp_email(email, subject, html)
