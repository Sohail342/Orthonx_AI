from datetime import datetime

from fastapi_users_db_sqlalchemy import (
    SQLAlchemyBaseUserTableUUID,
)
from sqlalchemy import (
    DateTime,
    Enum as SAEnum,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base_class import Base
from app.schemas.user import UserType


class User(SQLAlchemyBaseUserTableUUID, Base):
    __tablename__ = "user"

    email: Mapped[str] = mapped_column(
        String(length=320), unique=True, index=True, nullable=True
    )
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    phone_number: Mapped[str | None] = mapped_column(
        String, nullable=True, unique=True, index=True
    )
    user_type: Mapped[UserType] = mapped_column(
        SAEnum(UserType, name="user_type_enum"),
        default=UserType.USER,
        nullable=False,
    )

    credits: Mapped[int] = mapped_column(Integer, default=200, nullable=False)
    last_credit_reset_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), nullable=False
    )

    create_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=True,
    )

    # relationship → DiagnosisRecord
    diagnosis_records: Mapped[list["DiagnosisRecord"]] = relationship(  # noqa
        "DiagnosisRecord",
        back_populates="user",
        cascade="all, delete-orphan",
        foreign_keys="DiagnosisRecord.user_id",
    )

    # Relationships for appointments
    patient_appointments: Mapped[list["Appointment"]] = relationship(  # noqa
        "Appointment",
        back_populates="patient",
        foreign_keys="Appointment.patient_id",
        cascade="all, delete-orphan",
    )
    doctor_appointments: Mapped[list["Appointment"]] = relationship(  # noqa
        "Appointment",
        back_populates="doctor",
        foreign_keys="Appointment.doctor_id",
        cascade="all, delete-orphan",
    )


class SystemConfiguration(Base):
    __tablename__ = "system_configuration"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    value: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)

    create_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=True,
    )
