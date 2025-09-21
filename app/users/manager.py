from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi_users import BaseUserManager, UUIDIDMixin
from fastapi_users.exceptions import UserNotExists
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.session import get_db
from app.models.users import User
from app.users.dependencies import get_user_db
from app.utils.users import get_by_phone_no
from app.workers.tasks import send_password_reset_email, send_verification_request

SECRET = settings.SECRET_KEY


class UserManager(UUIDIDMixin, BaseUserManager[User, str]):
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

    def __init__(self, user_db, db: AsyncSession):
        super().__init__(user_db)
        self.db = db

    async def on_after_request_verify(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        """Send varification mail in background"""
        send_verification_request.delay(user.email, user.first_name, token)

    async def on_after_forgot_password(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        print(f"User {user.id} has forgot their password. Reset token: {token}")
        send_password_reset_email.delay(user.email, user.first_name, token)

    async def authenticate(
        self, credentials: OAuth2PasswordRequestForm
    ) -> Optional[User]:
        # Try by phone number
        user = await get_by_phone_no(credentials.username, self.db)

        if not user:
            # fallback: try normal email login
            try:
                return await super().authenticate(credentials)
            except UserNotExists:
                return None

        # Verify password (reuse FastAPI-Users’ helper)
        verified, updated_password_hash = self.password_helper.verify_and_update(
            credentials.password, user.hashed_password
        )
        if not verified:
            return None

        # Upgrade hash if necessary
        if updated_password_hash is not None:
            await self.user_db.update(user, {"hashed_password": updated_password_hash})

        return user

    async def create(self, user_create, safe=False, request=None):
        if user_create.phone_number:
            existing_user = await get_by_phone_no(user_create.phone_number, self.db)
            if existing_user is not None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="User already exists with this Phone number",
                )
        if user_create.email:
            existing_user = await self.user_db.get_by_email(user_create.email)
            if existing_user is not None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="User already exists with this Email address",
                )

        return await super().create(user_create, safe, request)


async def get_user_manager(
    user_db=Depends(get_user_db), db: AsyncSession = Depends(get_db)
):
    yield UserManager(user_db, db)
