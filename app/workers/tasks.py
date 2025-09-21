"""Celery tasks for background processing."""

import asyncio

from celery import shared_task
from fastapi_mail import FastMail, MessageSchema, MessageType

from app.core.config import settings
from app.utils.mail_config import conf

fm = FastMail(conf)


@shared_task
def send_verification_request(email: str, name: str, token: str) -> None:
    """Send email in a synchronous way (for Celery tasks)."""
    verify_url = (
        f"https://{settings.DOMAIN}/auth/verify/email?token={token}"
        if settings.ENVIRONMENT == "production"
        else f"http://127.0.0.1:8000/auth/verify/email?token={token}"
    )

    subject = "Verify your account"
    html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
            <title>Email Verification</title>
        </head>
        <body style="margin:0; padding:0; font-family: Arial, sans-serif; background-color:#f9fafb; color:#111827;">
            <table role="presentation" cellspacing="0" cellpadding="0" border="0" style="width:100%; border-collapse:collapse;">
            <tr>
                <td align="center" style="padding:2rem;">
                <table role="presentation" cellspacing="0" cellpadding="0" border="0" style="max-width:600px; width:100%; background-color:#ffffff; border-radius:8px; box-shadow:0 1px 2px rgba(0,0,0,0.1); overflow:hidden;">
                    <!-- Header -->
                    <tr>
                    <td align="center" style="background-color:#4f46e5; padding:24px;">
                        <h1 style="margin:0; font-size:24px; font-weight:600; color:#ffffff;">Welcome to MyApp 🎉</h1>
                    </td>
                    </tr>

                    <!-- Body -->
                    <tr>
                    <td style="padding:32px; text-align:left;">
                        <p style="font-size:16px; color:#374151; margin:0 0 24px 0;">
                        Hi <strong>{name}</strong>,
                        </p>
                        <p style="font-size:16px; color:#374151; margin:0 0 32px 0;">
                        Thanks for signing up! Please verify your account by clicking the button below:
                        </p>

                        <!-- Button - Using table method for better compatibility -->
                        <table role="presentation" cellspacing="0" cellpadding="0" border="0" style="margin:0 auto 32px auto;">
                        <tr>
                            <td style="border-radius:8px; background-color:#4f46e5;">
                            <a href="{verify_url}" target="_blank" style="
                                display:block;
                                background-color:#4f46e5;
                                color:#ffffff !important;
                                font-weight:600;
                                padding:12px 24px;
                                border-radius:8px;
                                text-decoration:none;
                                font-size:16px;
                                font-family: Arial, sans-serif;
                                border:none;
                                text-align:center;
                            ">
                                Verify Account
                            </a>
                            </td>
                        </tr>
                        </table>

                        <!-- Fallback link -->
                        <p style="font-size:14px; color:#6b7280; margin:0 0 16px 0; text-align:center;">
                        If the button doesn't work, copy and paste this link into your browser:
                        </p>
                        <p style="font-size:14px; color:#4f46e5; margin:0 0 24px 0; text-align:center; word-break:break-all;">
                        <a href="{verify_url}" style="color:#4f46e5;">{verify_url}</a>
                        </p>

                        <p style="font-size:14px; color:#6b7280; margin:0;">
                        If you did not create this account, you can safely ignore this email.
                        </p>
                    </td>
                    </tr>

                    <!-- Footer -->
                    <tr>
                    <td align="center" style="background-color:#f3f4f6; padding:16px;">
                        <p style="font-size:12px; color:#9ca3af; margin:0;">
                        &copy; {settings.PROJECT_NAME} {settings.PROJECT_YEAR}. All rights reserved.
                        </p>
                    </td>
                    </tr>
                </table>
                </td>
            </tr>
            </table>
        </body>
        </html>
        """

    message = MessageSchema(
        subject=subject,
        recipients=[email],
        body=html,
        subtype=MessageType.html,
    )

    asyncio.run(fm.send_message(message))


@shared_task
def send_password_reset_email(email: str, name: str, token: str):
    reset_url = (
        f"https://{settings.DOMAIN}/auth/reset-password?token={token}"
        if settings.ENVIRONMENT == "production"
        else f"http://127.0.0.1:8000/auth/reset-password?token={token}"
    )

    subject = "Reset Your Password"
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>Password Reset</title>
    </head>
    <body style="font-family: Arial, sans-serif; background-color:#f9fafb; color:#111827;">
        <table style="max-width:600px;margin:auto;background:#fff;border-radius:8px;padding:24px;">
            <tr>
                <td style="text-align:center;">
                    <h1 style="color:#4f46e5;">Reset Your Password</h1>
                    <p>Hello <strong>{name}</strong>,</p>
                    <p>You requested a password reset. Click the button below to set a new password:</p>
                    <a href="{reset_url}" style="
                        display:inline-block;
                        background-color:#4f46e5;
                        color:#fff;
                        padding:12px 24px;
                        border-radius:8px;
                        text-decoration:none;
                        font-weight:bold;
                        margin: 16px 0;
                    ">
                        Reset Password
                    </a>
                    <p>If you didn’t request this, just ignore this email.</p>
                    <p style="font-size:12px;color:#6b7280;">Or copy this link: <br>
                    <a href="{reset_url}" style="color:#4f46e5;">{reset_url}</a></p>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """

    message = MessageSchema(
        subject=subject,
        recipients=[email],
        body=html,
        subtype=MessageType.html,
    )
    asyncio.run(fm.send_message(message))
