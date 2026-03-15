from typing import Optional

from pydantic import BaseModel


class SuperadminDashboardResponse(BaseModel):
    role: str = "superadmin"
    total_users: int
    total_doctors: int
    total_diagnoses: int


class DoctorDashboardResponse(BaseModel):
    role: str = "doctor"
    total_diagnoses: int
    associated_hospitals: int
    doctor_name: Optional[str] = None
