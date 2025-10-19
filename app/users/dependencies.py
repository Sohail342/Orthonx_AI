from fastapi import Depends
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.models.users import User


async def get_user_db(
    session: AsyncSession = Depends(get_db),
) -> SQLAlchemyUserDatabase:
    yield SQLAlchemyUserDatabase(session, User)
