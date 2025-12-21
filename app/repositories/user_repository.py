from typing import Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.users import DiagnosisRecord, User


class UserRepository:
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    async def get_user_by_email(self, email: str) -> Optional[User]:
        stmt = select(User).where(User.email == email)
        result = await self.db_session.execute(stmt)
        user: Optional[User] = result.scalar_one_or_none()
        return user

    async def update_user_type(self, user_id: UUID, new_type: str) -> Optional[User]:
        stmt = select(User).where(User.id == user_id)
        result = await self.db_session.execute(stmt)
        user: Optional[User] = result.unique().scalar_one_or_none()

        if user is None:
            return None

        user.user_type = new_type
        self.db_session.add(user)
        await self.db_session.commit()
        await self.db_session.refresh(user)

        return user

    async def get_all_diagnosis_records(
        self,
        user_id: UUID,
        offset: int,
        limit: int,
    ) -> tuple[list[DiagnosisRecord], int]:
        # Total count
        count_stmt = select(func.count()).where(DiagnosisRecord.user_id == user_id)
        total = await self.db_session.scalar(count_stmt)

        # Paginated records
        stmt = (
            select(DiagnosisRecord)
            .where(DiagnosisRecord.user_id == user_id)
            .order_by(DiagnosisRecord.timestamp.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.db_session.execute(stmt)
        records = result.scalars().all()

        return list(records), total

    async def create_diagnosis_record(
        self,
        user_id: UUID,
        diagnosis_data: Optional[dict] = None,
        public_id: Optional[str] = None,
        uploaded_image_url: Optional[str] = None,
        result_image_url: Optional[str] = None,
        explanation_image_url: Optional[str] = None,
        gradcam_image_url: Optional[str] = None,
        report_url: Optional[str] = None,
    ) -> DiagnosisRecord:
        new_record = DiagnosisRecord(
            user_id=user_id,
            diagnosis_data=diagnosis_data,
            public_id=public_id,
            uploaded_image_url=uploaded_image_url,
            result_image_url=result_image_url,
            explanation_image_url=explanation_image_url,
            gradcam_image_url=gradcam_image_url,
            report_url=report_url,
        )
        self.db_session.add(new_record)
        await self.db_session.commit()
        await self.db_session.refresh(new_record)
        return new_record

    async def delete_diagnosis_record(
        self,
        record_id: int,
        user_id: Optional[UUID] = None,
    ) -> Optional[DiagnosisRecord]:
        stmt = select(DiagnosisRecord).where(
            DiagnosisRecord.id == record_id, DiagnosisRecord.user_id == user_id
        )
        result = await self.db_session.execute(stmt)
        record: Optional[DiagnosisRecord] = result.unique().scalar_one_or_none()

        if record is None:
            return None

        await self.db_session.delete(record)
        await self.db_session.commit()
        return record
