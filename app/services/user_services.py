import math
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.user_repository import UserRepository
from app.schemas.user import UserDiagnosisHistoryRequest


class UserServices:
    """User Services"""

    def __init__(self, db: AsyncSession):
        self.user_repository = UserRepository(db)

    async def get_detection_history(
        self,
        user_id: UUID,
        page: int = 1,
        page_size: int = 10,
    ) -> dict:
        offset = (page - 1) * page_size

        records, total = await self.user_repository.get_all_diagnosis_records(
            user_id=user_id,
            offset=offset,
            limit=page_size,
        )

        return {
            "items": [
                UserDiagnosisHistoryRequest.model_validate(record) for record in records
            ],
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "pages": math.ceil(total / page_size) if page_size else 0,
            },
        }

    async def delete_diagnosis_record(
        self,
        record_id: int,
        user_id: UUID,
    ) -> bool:
        try:
            await self.user_repository.delete_diagnosis_record(
                record_id=record_id,
                user_id=user_id,
            )
            return True
        except Exception:
            return False
