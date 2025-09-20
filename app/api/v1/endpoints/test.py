"""Test endpoints."""

from fastapi import APIRouter, Depends

from app.core.users import current_active_user

router = APIRouter()


@router.get("/test", dependencies=[Depends(current_active_user)])
async def test():
    return {"message": "Hello World from Sohail!"}
