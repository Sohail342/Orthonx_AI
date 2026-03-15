from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Enum as SAEnum, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base_class import Base
from app.models.users import User
from app.schemas.appointment import AppointmentStatus


class Appointment(Base):
    __tablename__ = "appointment"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)

    patient_id: Mapped[UUID] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    doctor_id: Mapped[UUID] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    appointment_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    time_slot: Mapped[str] = mapped_column(String, nullable=False)

    status: Mapped[AppointmentStatus] = mapped_column(
        SAEnum(AppointmentStatus, name="appointment_status_enum"),
        default=AppointmentStatus.PENDING,
        nullable=False,
    )

    patient_notes: Mapped[str | None] = mapped_column(String, nullable=True)
    doctor_notes: Mapped[str | None] = mapped_column(String, nullable=True)

    create_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    patient: Mapped["User"] = relationship(
        "User", back_populates="patient_appointments", foreign_keys=[patient_id]
    )
    doctor: Mapped["User"] = relationship(
        "User", back_populates="doctor_appointments", foreign_keys=[doctor_id]
    )
