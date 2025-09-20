from fastapi import Depends
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase

from app.database.session import get_db
from app.models.users import User


async def get_user_db(session=Depends(get_db)):
    yield SQLAlchemyUserDatabase(session, User)
