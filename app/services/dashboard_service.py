from fastapi import HTTPException

from app.models.users import User
from app.repositories.dashboard_repository import DashboardRepository
from app.schemas.dashboard import SuperadminDashboardResponse


class DashboardService:
    def __init__(self, repository: DashboardRepository):
        self.repository = repository

    async def get_superadmin_dashboard(
        self, current_user: User
    ) -> SuperadminDashboardResponse:
        if not current_user.is_superuser:
            raise HTTPException(status_code=403, detail="Not authorized")
        stats = await self.repository.get_superadmin_stats()
        return SuperadminDashboardResponse(**stats)
