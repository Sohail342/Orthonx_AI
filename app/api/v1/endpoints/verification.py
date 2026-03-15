"""Doctor Verification API endpoints."""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.users import current_active_user, current_superuser
from app.database.session import get_db
from app.models.users import User
from app.models.verification import DoctorVerification
from app.schemas.doctor import VerificationStatus

router = APIRouter()


class VerificationSubmit(BaseModel):
    full_name: str
    specialty: str
    license_number: str
    hospital_or_clinic: Optional[str] = None
    additional_notes: Optional[str] = None


class VerificationReview(BaseModel):
    status: str  # "approved" | "rejected"
    admin_notes: Optional[str] = None


def _serialize(v: DoctorVerification) -> dict:
    return {
        "id": v.id,
        "doctor_id": str(v.doctor_id),
        "full_name": v.full_name,
        "specialty": v.specialty,
        "license_number": v.license_number,
        "hospital_or_clinic": v.hospital_or_clinic,
        "status": v.status,
        "admin_notes": v.admin_notes,
        "additional_notes": v.additional_notes,
        "submitted_at": v.submitted_at.isoformat() if v.submitted_at else None,
        "reviewed_at": v.reviewed_at.isoformat() if v.reviewed_at else None,
    }


@router.post("/submit")
async def submit_verification(
    data: VerificationSubmit,
    current_user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Doctor submits credentials for admin review."""
    if current_user.user_type != "doctor":
        raise HTTPException(
            status_code=403, detail="Only doctors can submit verification"
        )

    # Check if already submitted
    existing = await db.execute(
        select(DoctorVerification).where(
            DoctorVerification.doctor_id == current_user.id
        )
    )
    record = existing.scalars().first()

    if record:
        if record.status == VerificationStatus.APPROVED:
            raise HTTPException(status_code=400, detail="Already verified")
        # Allow re-submission if rejected
        record.full_name = data.full_name
        record.specialty = data.specialty
        record.license_number = data.license_number
        record.hospital_or_clinic = data.hospital_or_clinic
        record.additional_notes = data.additional_notes
        record.status = VerificationStatus.PENDING
        record.admin_notes = None
        db.add(record)
    else:
        record = DoctorVerification(
            doctor_id=current_user.id,
            full_name=data.full_name,
            specialty=data.specialty,
            license_number=data.license_number,
            hospital_or_clinic=data.hospital_or_clinic,
            additional_notes=data.additional_notes,
        )
        db.add(record)

    await db.commit()
    await db.refresh(record)
    return _serialize(record)


@router.get("/status")
async def get_verification_status(
    current_user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the current doctor's verification status."""
    result = await db.execute(
        select(DoctorVerification).where(
            DoctorVerification.doctor_id == current_user.id
        )
    )
    record = result.scalars().first()
    if not record:
        return {"status": "not_submitted"}
    return _serialize(record)


@router.get("/all")
async def get_all_verifications(
    current_user: User = Depends(current_superuser),
    db: AsyncSession = Depends(get_db),
    status: Optional[str] = None,
):
    """List all doctor verifications, optionally filtered by status."""
    q = select(DoctorVerification)
    if status:
        q = q.where(DoctorVerification.status == status)
    q = q.order_by(DoctorVerification.submitted_at.desc())
    result = await db.execute(q)
    records = result.scalars().all()
    return [_serialize(r) for r in records]


@router.put("/{verification_id}/review")
async def review_verification(
    verification_id: int,
    data: VerificationReview,
    current_user: User = Depends(current_superuser),
    db: AsyncSession = Depends(get_db),
):
    """Approve or reject a doctor verification request."""
    # The validation is now implicitly handled by the Pydantic/Enum integration if we updated the schema,
    # but since VerificationReview uses 'str', we check manually or update the schema.
    # Let's update the manual check for now to use the Enum values.
    try:
        new_status = VerificationStatus(data.status)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Status must be 'approved' or 'rejected'"
        )

    result = await db.execute(
        select(DoctorVerification).where(DoctorVerification.id == verification_id)
    )
    record = result.scalars().first()
    if not record:
        raise HTTPException(status_code=404, detail="Verification not found")

    record.status = new_status
    record.admin_notes = data.admin_notes
    record.reviewed_at = datetime.now(timezone.utc)
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return _serialize(record)
