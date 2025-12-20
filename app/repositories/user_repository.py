from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.users import User


async def update_user_type(
    user_id: str, new_type: str, db: AsyncSession
) -> Optional[User]:
    stmt = select(User).where(User.id == user_id)
    result = await db.execute(stmt)
    user: Optional[User] = result.scalar_one_or_none()

    if user is None:
        return None

    user.user_type = new_type
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return user
