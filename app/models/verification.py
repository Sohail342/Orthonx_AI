from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Enum as SAEnum, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base_class import Base
from app.models.users import User
from app.schemas.doctor import VerificationStatus


class DoctorVerification(Base):
    """Stores doctor credential documents and verification status."""

    __tablename__ = "doctor_verification"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    doctor_id: Mapped[UUID] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # Submission fields
    full_name: Mapped[str] = mapped_column(String, nullable=False)
    specialty: Mapped[str] = mapped_column(String, nullable=False)
    license_number: Mapped[str] = mapped_column(String, nullable=False)
    hospital_or_clinic: Mapped[str | None] = mapped_column(String, nullable=True)
    document_url: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # uploaded credential doc
    additional_notes: Mapped[str | None] = mapped_column(String, nullable=True)

    # Admin review fields
    status: Mapped[VerificationStatus] = mapped_column(
        SAEnum(VerificationStatus, name="verification_status_enum"),
        default=VerificationStatus.PENDING,
        nullable=False,
        index=True,
    )
    admin_notes: Mapped[str | None] = mapped_column(String, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=True,
    )

    # Relationships
    doctor: Mapped["User"] = relationship("User", foreign_keys=[doctor_id])
