from typing import Optional, cast

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.users import User


async def get_by_phone_no(phone_no: str, db: AsyncSession) -> Optional[User]:
    """Get User by phone number

    Args:
        phone_no (str): User's Phone Number
        db (AsyncSession): Asyn Database Session
    """
    result = await db.execute(select(User).where(User.phone_number == phone_no))
    return cast(Optional[User], result.scalar_one_or_none())
