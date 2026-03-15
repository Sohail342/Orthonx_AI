from datetime import datetime
from uuid import UUID

from sqlalchemy import JSON, DateTime, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base_class import Base
from app.models.users import User


class DiagnosisRecord(Base):
    __tablename__ = "diagnosis_record"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    public_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    uploaded_image_url: Mapped[str] = mapped_column(String, nullable=False)
    result_image_url: Mapped[str] = mapped_column(String, nullable=False)
    explanation_image_url: Mapped[str] = mapped_column(String, nullable=False)
    gradcam_image_url: Mapped[str] = mapped_column(String, nullable=False)
    report_url: Mapped[str] = mapped_column(String, nullable=False)
    diagnosis_data: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    review_status: Mapped[str] = mapped_column(String, default="none", nullable=True)
    reviewer_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
    )
    review_notes: Mapped[str | None] = mapped_column(String, nullable=True)

    create_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=True,
    )

    # relationship → User
    user: Mapped["User"] = relationship(
        "User", back_populates="diagnosis_records", foreign_keys=[user_id]
    )
    reviewer: Mapped["User"] = relationship("User", foreign_keys=[reviewer_id])
