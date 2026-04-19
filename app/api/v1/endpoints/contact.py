"""Contact Us API endpoints.

Public endpoint – no authentication required.
Admin-only list/mark-resolved endpoints are protected.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.users import current_active_user
from app.database.session import get_db
from app.models.contact import ContactMessage
from app.models.users import User
from app.schemas.contact import (
    ContactMessageCreate,
    ContactMessageRead,
    ContactMessageResponse,
)

router = APIRouter()


@router.post("/", response_model=ContactMessageResponse, status_code=201)
async def submit_contact_message(
    payload: ContactMessageCreate,
    db: AsyncSession = Depends(get_db),
):
    """Anyone (guest or authenticated) can submit a contact message."""
    new_message = ContactMessage(
        name=payload.name,
        email=payload.email,
        subject=payload.subject,
        message=payload.message,
    )
    db.add(new_message)
    await db.commit()
    return ContactMessageResponse(
        message="Thank you for reaching out! We will get back to you soon."
    )


@router.get("/", response_model=List[ContactMessageRead])
async def list_contact_messages(
    current_user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Superadmin only: retrieve all submitted contact messages."""
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not authorized.")

    stmt = select(ContactMessage).order_by(ContactMessage.created_at.desc())
    result = await db.execute(stmt)
    return result.scalars().all()


@router.patch("/{message_id}/read", response_model=ContactMessageRead)
async def mark_as_read(
    message_id: int,
    current_user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Superadmin only: mark a contact message as read."""
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not authorized.")

    stmt = select(ContactMessage).where(ContactMessage.id == message_id)
    result = await db.execute(stmt)
    msg = result.scalar_one_or_none()
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found.")

    msg.is_read = True
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg


@router.patch("/{message_id}/resolve", response_model=ContactMessageRead)
async def mark_as_resolved(
    message_id: int,
    current_user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Superadmin only: mark a contact message as resolved."""
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not authorized.")

    stmt = select(ContactMessage).where(ContactMessage.id == message_id)
    result = await db.execute(stmt)
    msg = result.scalar_one_or_none()
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found.")

    msg.is_read = True
    msg.is_resolved = True
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg
