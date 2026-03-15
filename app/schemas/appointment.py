import enum
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class AppointmentStatus(str, enum.Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class AppointmentBase(BaseModel):
    doctor_id: UUID
    appointment_date: datetime
    time_slot: str
    patient_notes: Optional[str] = None


class AppointmentCreate(AppointmentBase):
    pass


class AppointmentUpdate(BaseModel):
    status: Optional[AppointmentStatus] = None
    doctor_notes: Optional[str] = None


class UserShort(BaseModel):
    id: UUID
    email: str
    name: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class AppointmentRead(BaseModel):
    id: UUID
    patient_id: UUID
    doctor_id: UUID
    appointment_date: datetime
    time_slot: str
    status: AppointmentStatus
    patient_notes: Optional[str] = None
    doctor_notes: Optional[str] = None
    create_at: datetime

    patient: Optional[UserShort] = None
    doctor: Optional[UserShort] = None

    model_config = ConfigDict(from_attributes=True)


class AppointmentList(BaseModel):
    total: int
    items: List[AppointmentRead]
