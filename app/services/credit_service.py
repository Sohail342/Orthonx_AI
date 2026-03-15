from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.users import SystemConfiguration, User

# Fixed defaults if not found in db
DEFAULTS = {
    "DEFAULT_CREDITS": 200,
    "UPLOAD_IMAGE_COST": 10,
    "REQUEST_REVIEW_COST": 20,
    "DOCTOR_REVIEW_EARNING": 20,
    "APPOINTMENT_BOOKING_COST": 100,
    "APPOINTMENT_COMPLETION_EARNING": 100,
}


class CreditService:
    @staticmethod
    async def get_config_value(session: AsyncSession, key: str) -> int:
        """Fetches a configuration value from the db, creating it with default if it doesn't exist."""
        result = await session.execute(
            select(SystemConfiguration).where(SystemConfiguration.key == key)
        )
        config = result.scalars().first()
        if config:
            return config.value

        # Create it if missing
        default_val = DEFAULTS.get(key, 0)
        new_config = SystemConfiguration(
            key=key, value=default_val, description=f"Default value for {key}"
        )
        session.add(new_config)
        await session.commit()
        return default_val

    @staticmethod
    async def check_and_reset_monthly_credits(
        user: User, session: AsyncSession
    ) -> None:
        """
        Lazily evaluate if the user's credits need restoring for the new month.
        If the user's `last_credit_reset_date` is from a previous month (or year),
        set their credits back to `DEFAULT_CREDITS` and update the reset timestamp.
        """
        now = datetime.now(user.last_credit_reset_date.tzinfo)
        last_reset = user.last_credit_reset_date

        # Check if the current month/year is different from the last reset month/year
        if now.year > last_reset.year or (
            now.year == last_reset.year and now.month > last_reset.month
        ):
            default_credits = await CreditService.get_config_value(
                session, "DEFAULT_CREDITS"
            )

            user.credits = default_credits
            user.last_credit_reset_date = now

            session.add(user)
            await session.commit()
            await session.refresh(user)
