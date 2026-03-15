from typing import Any, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.users import current_active_user, current_superuser
from app.database.session import get_db
from app.models.appointment import Appointment
from app.models.users import User
from app.models.verification import DoctorVerification
from app.schemas.appointment import (
    AppointmentCreate,
    AppointmentList,
    AppointmentRead,
    AppointmentStatus,
    AppointmentUpdate,
)
from app.schemas.doctor import VerificationStatus
from app.schemas.user import UserRead, UserType
from app.services.credit_service import CreditService

router = APIRouter()


@router.get("/doctors", response_model=List[UserRead])
async def list_doctors(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(current_active_user),
) -> Any:
    """List all available doctors for booking."""
    # Only return users with user_type='doctor' who are active and formally approved
    query = (
        select(User)
        .join(DoctorVerification, User.id == DoctorVerification.doctor_id)
        .where(
            User.user_type == UserType.DOCTOR,
            User.is_active,
            DoctorVerification.status == VerificationStatus.APPROVED,
        )
    )
    result = await db.execute(query)
    return result.scalars().all()


@router.post("/", response_model=AppointmentRead)
async def book_appointment(
    *,
    db: AsyncSession = Depends(get_db),
    appointment_in: AppointmentCreate,
    current_user: User = Depends(current_active_user),
) -> Any:
    """Book a new appointment."""
    # Check if doctor exists and is actually a doctor
    doctor_result = await db.execute(
        select(User).where(User.id == appointment_in.doctor_id)
    )
    doctor = doctor_result.scalars().first()
    if not doctor or doctor.user_type != UserType.DOCTOR:
        raise HTTPException(status_code=404, detail="Doctor not found")

    # Credit logic: Check and deduct credits
    booking_cost = await CreditService.get_config_value(db, "APPOINTMENT_BOOKING_COST")
    if current_user.credits < booking_cost:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient credits. Booking an appointment costs {booking_cost} credits.",
        )

    appointment = Appointment(
        **appointment_in.model_dump(),
        patient_id=current_user.id,
        status=AppointmentStatus.PENDING,
    )

    # Deduct credits
    current_user.credits -= booking_cost
    db.add(current_user)
    db.add(appointment)

    await db.commit()
    await db.refresh(appointment)

    # Load relationships for response
    result = await db.execute(
        select(Appointment)
        .where(Appointment.id == appointment.id)
        .options(selectinload(Appointment.patient), selectinload(Appointment.doctor))
    )
    return result.scalars().first()


@router.get("/me/patient", response_model=AppointmentList)
async def get_my_appointments(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(current_active_user),
    skip: int = 0,
    limit: int = 20,
) -> Any:
    """Get appointments for the current patient."""
    query = (
        select(Appointment)
        .options(selectinload(Appointment.doctor))
        .where(Appointment.patient_id == current_user.id)
        .order_by(desc(Appointment.appointment_date))
    )

    # Count
    count_query = select(func.count()).where(Appointment.patient_id == current_user.id)
    total = (await db.execute(count_query)).scalar() or 0

    # Results
    result = await db.execute(query.offset(skip).limit(limit))
    items = result.scalars().all()

    return {"total": total, "items": items}


@router.get("/me/doctor", response_model=AppointmentList)
async def get_doctor_appointments(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(current_active_user),
    skip: int = 0,
    limit: int = 20,
) -> Any:
    """Get appointments for the current doctor."""
    if current_user.user_type != UserType.DOCTOR:
        raise HTTPException(
            status_code=403, detail="Only doctors can access this endpoint"
        )

    query = (
        select(Appointment)
        .options(selectinload(Appointment.patient))
        .where(Appointment.doctor_id == current_user.id)
        .order_by(desc(Appointment.appointment_date))
    )

    # Count
    count_query = select(func.count()).where(Appointment.doctor_id == current_user.id)
    total = (await db.execute(count_query)).scalar() or 0

    # Results
    result = await db.execute(query.offset(skip).limit(limit))
    items = result.scalars().all()

    return {"total": total, "items": items}


@router.patch("/{appointment_id}", response_model=AppointmentRead)
async def update_appointment(
    *,
    db: AsyncSession = Depends(get_db),
    appointment_id: UUID,
    appointment_in: AppointmentUpdate,
    current_user: User = Depends(current_active_user),
) -> Any:
    """Update appointment status or notes."""
    result = await db.execute(
        select(Appointment).where(Appointment.id == appointment_id)
    )
    appointment = result.scalars().first()
    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")

    # Permissions check: Only doctor of this appointment or superadmin can update
    if not current_user.is_superuser and appointment.doctor_id != current_user.id:
        raise HTTPException(
            status_code=403, detail="Not authorized to update this appointment"
        )

    update_data = appointment_in.model_dump(exclude_unset=True)

    # Check if we are marking it as completed to reward credits
    reward_doctor = False
    if (
        "status" in update_data
        and update_data["status"] == AppointmentStatus.COMPLETED
        and appointment.status != AppointmentStatus.COMPLETED
    ):
        reward_doctor = True

    for field, value in update_data.items():
        setattr(appointment, field, value)

    db.add(appointment)

    if reward_doctor:
        # Load doctor to reward
        doctor_res = await db.execute(
            select(User).where(User.id == appointment.doctor_id)
        )
        doctor = doctor_res.scalars().first()
        if doctor:
            earning = await CreditService.get_config_value(
                db, "APPOINTMENT_COMPLETION_EARNING"
            )
            doctor.credits += earning
            db.add(doctor)

    await db.commit()

    # Reload with relationships for the response
    result = await db.execute(
        select(Appointment)
        .where(Appointment.id == appointment.id)
        .options(selectinload(Appointment.patient), selectinload(Appointment.doctor))
    )
    return result.scalars().first()


@router.get("/all", response_model=AppointmentList)
async def get_all_appointments(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(current_superuser),
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """Get all appointments (admin only)."""
    query = (
        select(Appointment)
        .options(selectinload(Appointment.patient), selectinload(Appointment.doctor))
        .order_by(desc(Appointment.create_at))
    )

    count_query = select(func.count()).select_from(Appointment)
    total = (await db.execute(count_query)).scalar() or 0

    result = await db.execute(query.offset(skip).limit(limit))
    return {"total": total, "items": result.scalars().all()}
