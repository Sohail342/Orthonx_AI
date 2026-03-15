from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.users import current_active_user
from app.database.session import get_db
from app.models.diagnosis import DiagnosisRecord
from app.models.users import User
from app.schemas.user import ReviewSubmitRequest, UserDiagnosisHistoryRequest, UserType

router = APIRouter()


@router.get("/{record_id}", response_model=UserDiagnosisHistoryRequest)
async def get_record(
    record_id: int,
    current_user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(DiagnosisRecord).where(DiagnosisRecord.id == record_id)
    result = await db.execute(stmt)
    record = result.scalar_one_or_none()

    if not record:
        raise HTTPException(status_code=404, detail="Diagnosis record not found.")

    if current_user.user_type == UserType.USER and record.user_id != current_user.id:
        raise HTTPException(
            status_code=403, detail="Not authorized to view this record."
        )

    return record


@router.post("/{record_id}/request-review")
async def request_review(
    record_id: int,
    current_user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user.user_type != UserType.USER:
        raise HTTPException(
            status_code=403, detail="Only patients can request reviews."
        )

    # Get the record
    stmt = select(DiagnosisRecord).where(
        DiagnosisRecord.id == record_id, DiagnosisRecord.user_id == current_user.id
    )
    result = await db.execute(stmt)
    record = result.scalar_one_or_none()

    if not record:
        raise HTTPException(status_code=404, detail="Diagnosis record not found.")

    if record.review_status and record.review_status != "none":
        raise HTTPException(
            status_code=400, detail="Review already requested or completed."
        )

    from app.services.credit_service import CreditService

    await CreditService.check_and_reset_monthly_credits(current_user, db)
    cost = await CreditService.get_config_value(db, "REQUEST_REVIEW_COST")
    if current_user.credits < cost:
        raise HTTPException(
            status_code=402, detail=f"Insufficient credits. Required: {cost}"
        )

    # Update status
    record.review_status = "pending"
    current_user.credits -= cost

    db.add(record)
    db.add(current_user)
    await db.commit()
    return {"message": "Review requested successfully."}


@router.get("/reviews/pending", response_model=List[UserDiagnosisHistoryRequest])
async def get_pending_reviews(
    current_user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user.user_type != UserType.DOCTOR:
        raise HTTPException(
            status_code=403, detail="Only doctors can access pending reviews."
        )

    # For now, return all pending reviews globally. Could be filtered to hospitals the doctor is linked to.
    stmt = select(DiagnosisRecord).where(DiagnosisRecord.review_status == "pending")
    result = await db.execute(stmt)
    records = result.scalars().all()
    return records


@router.post("/{record_id}/submit-review")
async def submit_review(
    record_id: int,
    review_data: ReviewSubmitRequest,
    current_user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user.user_type != UserType.DOCTOR:
        raise HTTPException(status_code=403, detail="Only doctors can submit reviews.")

    # Get the record
    stmt = select(DiagnosisRecord).where(DiagnosisRecord.id == record_id)
    result = await db.execute(stmt)
    record = result.scalar_one_or_none()

    if not record:
        raise HTTPException(status_code=404, detail="Diagnosis record not found.")

    if record.review_status != "pending":
        raise HTTPException(status_code=400, detail="Record is not pending review.")

    from app.services.credit_service import CreditService

    await CreditService.check_and_reset_monthly_credits(current_user, db)
    earning = await CreditService.get_config_value(db, "DOCTOR_REVIEW_EARNING")

    # Update record
    record.review_status = "reviewed"
    record.reviewer_id = current_user.id
    record.review_notes = review_data.review_notes
    current_user.credits += earning

    db.add(record)
    db.add(current_user)
    await db.commit()

    return {"message": "Review submitted successfully."}
