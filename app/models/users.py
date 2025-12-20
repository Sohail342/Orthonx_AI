from fastapi_users_db_sqlalchemy import (
    SQLAlchemyBaseOAuthAccountTableUUID,
    SQLAlchemyBaseUserTableUUID,
)
from sqlalchemy import Enum as SAEnum, String
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
