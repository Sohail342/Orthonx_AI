"""User schemas for FastAPI-Users integration."""

import re
import uuid
from typing import Annotated, Literal

from fastapi_users import schemas
from pydantic import EmailStr, Field, field_validator, model_validator

PAKISTAN_MOBILE_REGEX = re.compile(r"^(?:\+92|0)3[0-9]{9}$")


class UserRead(schemas.BaseUser[uuid.UUID]):
    """User Read Schema"""

    email: str | None = None
    first_name: str | None = None
    phone_number: str | None = None

    @field_validator("phone_number")
    def validate_phone_number(cls, v) -> str:
        if v:
            return "".join([digit for digit in v])
        return None

    model_config = {"exclude_none": True}


class UserCreate(schemas.BaseUserCreate):
    """User Create Schema"""

    email: (
        Annotated[
            EmailStr, Field(example="indus@gmail.com", description="Your email Address")
        ]
        | None
    ) = None
    registration_type: Annotated[
        str,
        Literal["email", "phone"],
        Field(
            ...,
            description="Registration with: either 'email' or 'phone'",
            examples=["email", "phone"],
            exclude=True,
        ),
    ]
    first_name: str | None = None
    phone_number: (
        Annotated[
            str,
            Field(
                example="03428041928",
                description="Pakistani mobile number (e.g. 03001234567, +923001234567)",
            ),
        ]
        | None
    ) = None

    @field_validator("phone_number")
    def validate_phone_number(cls, v: str) -> str:
        if v and not PAKISTAN_MOBILE_REGEX.match(v):
            raise ValueError(
                "Invalid Pakistani mobile number. Use format 030XXXXXXXX or +9230XXXXXXXX"
            )
        return v

    @model_validator(mode="before")
    def validate_required_fields(cls, data: dict):
        reg_type = data.get("registration_type")

        if reg_type == "phone":
            if not data.get("phone_number"):
                raise ValueError(
                    "Phone number is required when registration_type='phone'"
                )
            # Ensure email is unset
            data["email"] = None

        elif reg_type == "email":
            if not data.get("email"):
                raise ValueError("Email is required when registration_type='email'")
            # Ensure phone_number is unset
            data["phone_number"] = None

        return data

    @field_validator("registration_type")
    def validate_registration_type(cls, v: str) -> str:
        if v not in ("phone", "email"):
            raise ValueError(
                "Invalid registration_type: must be either 'email' or 'phone'."
            )
        return v


class UserUpdate(schemas.BaseUserUpdate):
    """User Update Schema"""

    first_name: str | None = None
    phone_number: str | None = None
