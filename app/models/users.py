from fastapi_users_db_sqlalchemy import SQLAlchemyBaseUserTableUUID
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base_class import Base


class User(SQLAlchemyBaseUserTableUUID, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(
        String(length=320), unique=True, index=True, nullable=True
    )
    first_name: Mapped[str | None] = mapped_column(String, nullable=True)
    phone_number: Mapped[str | None] = mapped_column(
        String, nullable=True, unique=True, index=True
    )
