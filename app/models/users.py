from datetime import datetime
from uuid import UUID

from fastapi_users_db_sqlalchemy import (
    SQLAlchemyBaseOAuthAccountTableUUID,
    SQLAlchemyBaseUserTableUUID,
)
from sqlalchemy import JSON, DateTime, Enum as SAEnum, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base_class import Base
from app.schemas.user import UserType


class OAuthAccount(SQLAlchemyBaseOAuthAccountTableUUID, Base):
    pass


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
    oauth_accounts: Mapped[list[OAuthAccount]] = relationship(
        "OAuthAccount", lazy="joined"
    )

    # relationship → DiagnosisRecord
    diagnosis_records: Mapped[list["DiagnosisRecord"]] = relationship(
        "DiagnosisRecord", back_populates="user", cascade="all, delete-orphan"
    )


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

    # relationship → User
    user: Mapped["User"] = relationship("User", back_populates="diagnosis_records")
