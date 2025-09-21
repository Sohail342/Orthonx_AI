"""Test endpoints."""

from fastapi import APIRouter, Depends, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi_users.manager import BaseUserManager

from app.users.manager import get_user_manager

verify_router = APIRouter()

templates = Jinja2Templates(directory="app/templates")


@verify_router.get("/auth/verify/email", response_class=HTMLResponse)
async def verify_via_get(
    request: Request,
    token: str,
    user_manager: BaseUserManager = Depends(get_user_manager),
):
    """Custom Verify Endpoint with GET Method"""
    try:
        await user_manager.verify(token, request)
        return templates.TemplateResponse("verify_success.html", {"request": request})
    except Exception as e:
        return templates.TemplateResponse(
            "verify_failed.html", {"request": request, "detail": str(e)}
        )
        
@verify_router.get("/auth/reset-password", response_class=HTMLResponse)
async def reset_password_form(request: Request, token: str):
    return templates.TemplateResponse("reset_password.html", {"request": request, "token": token})

@verify_router.post("/auth/reset-password", response_class=HTMLResponse)
async def reset_password_submit(
    request: Request,
    token: str = Form(...),
    password: str = Form(...),
    password_confirm: str = Form(...),
    user_manager: BaseUserManager = Depends(get_user_manager),
):
    message = None
    error = None

    if password != password_confirm:
        error = "Passwords do not match."
    else:
        try:
            await user_manager.reset_password(token, password)
            message = "Your password has been reset successfully. You can now log in."
        except Exception:
            error = "Invalid or expired token. Please request a new password reset link."

    return templates.TemplateResponse(
        "reset_password.html",
        {
            "request": request,
            "token": token,  # Keep token in case they retry
            "message": message,
            "error": error,
        },
    )
