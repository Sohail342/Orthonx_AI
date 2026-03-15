from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.diagnosis import DiagnosisRecord
from app.models.users import User
from app.schemas.user import UserType


class DashboardRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_superadmin_stats(self) -> dict:
        total_users = await self.session.scalar(
            select(func.count(User.id)).where(User.user_type == UserType.USER)
        )

        total_doctors = await self.session.scalar(
            select(func.count(User.id)).where(User.user_type == UserType.DOCTOR)
        )
        total_diagnoses = await self.session.scalar(
            select(func.count(DiagnosisRecord.id))
        )

        return {
            "total_users": total_users or 0,
            "total_doctors": total_doctors or 0,
            "total_diagnoses": total_diagnoses or 0,
        }
