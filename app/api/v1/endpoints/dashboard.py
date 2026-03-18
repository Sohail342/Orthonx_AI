from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.users import current_active_user, current_superuser
from app.database.session import get_db
from app.models.users import User
from app.repositories.dashboard_repository import DashboardRepository
from app.schemas.dashboard import (
    DoctorDashboardResponse,
    SuperadminDashboardResponse,
)
from app.schemas.user import SystemConfigUpdate, UserCreditUpdate
from app.services.credit_service import CreditService
from app.services.dashboard_service import DashboardService

router = APIRouter()


def get_dashboard_service(db: AsyncSession = Depends(get_db)) -> DashboardService:
    repository = DashboardRepository(db)
    return DashboardService(repository)


@router.get("/superadmin", response_model=SuperadminDashboardResponse)
async def get_superadmin_dashboard(
    current_user: User = Depends(current_active_user),
    service: DashboardService = Depends(get_dashboard_service),
):
    return await service.get_superadmin_dashboard(current_user)


@router.get("/doctor", response_model=DoctorDashboardResponse)
async def get_doctor_dashboard(
    current_user: User = Depends(current_active_user),
    service: DashboardService = Depends(get_dashboard_service),
):
    return await service.get_doctor_dashboard(current_user)


@router.get("/doctor-stats")
async def get_doctor_stats(
    current_user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import func, select

    from app.models.diagnosis import DiagnosisRecord

    # Total AI detections by this doctor
    total_detections_result = await db.execute(
        select(func.count()).where(DiagnosisRecord.user_id == current_user.id)
    )
    total_detections = total_detections_result.scalar() or 0

    # Pending review requests (across all patients)
    pending_reviews_result = await db.execute(
        select(func.count()).where(DiagnosisRecord.review_status == "pending")
    )
    pending_reviews = pending_reviews_result.scalar() or 0

    # Reviews completed by this doctor (reviewer_id is the field name)
    reviews_done_result = await db.execute(
        select(func.count()).where(
            DiagnosisRecord.reviewer_id == current_user.id,
            DiagnosisRecord.review_status == "reviewed",
        )
    )
    reviews_done = reviews_done_result.scalar() or 0

    return {
        "total_detections": total_detections,
        "pending_reviews": pending_reviews,
        "reviews_completed": reviews_done,
        "credits": current_user.credits,
        "credits_last_reset": current_user.last_credit_reset_date.isoformat()
        if current_user.last_credit_reset_date
        else None,
    }


@router.get("/config")
async def get_system_config(
    current_user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_db),
):
    return {
        "default_credits": await CreditService.get_config_value(db, "DEFAULT_CREDITS"),
        "upload_image_cost": await CreditService.get_config_value(
            db, "UPLOAD_IMAGE_COST"
        ),
        "request_review_cost": await CreditService.get_config_value(
            db, "REQUEST_REVIEW_COST"
        ),
        "doctor_review_earning": await CreditService.get_config_value(
            db, "DOCTOR_REVIEW_EARNING"
        ),
        "appointment_booking_cost": await CreditService.get_config_value(
            db, "APPOINTMENT_BOOKING_COST"
        ),
        "appointment_completion_earning": await CreditService.get_config_value(
            db, "APPOINTMENT_COMPLETION_EARNING"
        ),
    }


@router.put("/config")
async def update_system_config(
    config_update: SystemConfigUpdate,
    current_user: User = Depends(current_superuser),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select

    from app.models.users import SystemConfiguration

    async def set_val(key, val):
        result = await db.execute(
            select(SystemConfiguration).where(SystemConfiguration.key == key)
        )
        cfg = result.scalars().first()
        if cfg:
            cfg.value = val
        else:
            db.add(SystemConfiguration(key=key, value=val))

    await set_val("DEFAULT_CREDITS", config_update.default_credits)
    await set_val("UPLOAD_IMAGE_COST", config_update.upload_image_cost)
    await set_val("REQUEST_REVIEW_COST", config_update.request_review_cost)
    await set_val("DOCTOR_REVIEW_EARNING", config_update.doctor_review_earning)
    await set_val("APPOINTMENT_BOOKING_COST", config_update.appointment_booking_cost)
    await set_val(
        "APPOINTMENT_COMPLETION_EARNING", config_update.appointment_completion_earning
    )
    await db.commit()
    return {"message": "Configuration updated successfully"}


@router.put("/users/{user_id}/credits")
async def update_user_credits(
    user_id: str,
    credit_update: UserCreditUpdate,
    current_user: User = Depends(current_superuser),
    db: AsyncSession = Depends(get_db),
):
    import uuid

    from fastapi import HTTPException
    from sqlalchemy import select

    try:
        uid = uuid.UUID(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid UUID")

    result = await db.execute(select(User).where(User.id == uid))
    target = result.scalars().first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    target.credits = credit_update.credits
    db.add(target)
    await db.commit()
    return {"message": "Credits updated successfully", "credits": target.credits}


@router.get("/users")
async def get_all_users(
    current_user: User = Depends(current_superuser),
    db: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 10,
    search: str = "",
):
    from sqlalchemy import func, or_, select

    # Build base query with optional search filter
    base_query = select(User)
    if search:
        search_term = f"%{search}%"
        base_query = base_query.where(
            or_(
                User.email.ilike(search_term),
                User.name.ilike(search_term),
            )
        )

    # Get total count
    count_result = await db.execute(
        select(func.count()).select_from(base_query.subquery())
    )
    total = count_result.scalar() or 0

    # Get paginated results
    paginated_query = base_query.offset(skip).limit(limit)
    result = await db.execute(paginated_query)
    users = result.scalars().all()

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "items": [
            {
                "id": str(u.id),
                "email": u.email,
                "is_active": u.is_active,
                "is_superuser": u.is_superuser,
                "is_verified": u.is_verified,
                "user_type": u.user_type,
                "name": u.name,
                "phone_number": u.phone_number,
                "credits": u.credits,
                "last_credit_reset_date": u.last_credit_reset_date.isoformat()
                if u.last_credit_reset_date
                else None,
            }
            for u in users
        ],
    }
