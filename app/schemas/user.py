"""User schemas for FastAPI-Users integration."""

import datetime
import re
import uuid
from enum import Enum
from typing import Literal, Optional, Type

from fastapi_users import schemas
from pydantic import EmailStr, Field, field_validator, model_validator
from typing_extensions import Annotated

PAKISTAN_MOBILE_REGEX = re.compile(r"^(?:\+92|0)3[0-9]{9}$")


class UserType(str, Enum):
    USER = "user"
    HOSPITAL = "hospital"
    DOCTOR = "doctor"


class UserRead(schemas.BaseUser[uuid.UUID]):
    """User Read Schema"""

    email: Optional[str] = None
    user_type: UserType = UserType.USER
    name: Optional[str] = None
    phone_number: Optional[str] = None

    @field_validator("phone_number")
    def validate_phone_number(cls: Type["UserRead"], v: Optional[str]) -> Optional[str]:
        if v:
            return "".join([digit for digit in v])
        return None

    model_config = {"exclude_none": True}


class UserCreate(schemas.BaseUserCreate):
    """User Create Schema"""

    email: Optional[
        Annotated[
            EmailStr, Field(example="indus@gmail.com", description="Your email Address")
        ]
    ] = None
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
    user_type: Annotated[
        UserType,
        Field(
            ...,
            description="User type (e.g user, doctor, or hospital)",
        ),
    ] = UserType.USER
    name: Optional[str] = None
    phone_number: Optional[
        Annotated[
            str,
            Field(
                example="03428041928",
                description="Pakistani mobile number (e.g. 03001234567, +923001234567)",
            ),
        ]
    ] = None

    @field_validator("phone_number")
    def validate_phone_number(
        cls: Type["UserCreate"], v: Optional[str]
    ) -> Optional[str]:
        if v and not PAKISTAN_MOBILE_REGEX.match(v):
            raise ValueError(
                "Invalid Pakistani mobile number. Use format 030XXXXXXXX or +9230XXXXXXXX"
            )
        return v

    @model_validator(mode="before")
    def validate_required_fields(cls: Type["UserCreate"], data: dict) -> dict:
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

        else:
            raise ValueError("registration_type must be either 'email' or 'phone'")
        return data

    @field_validator("registration_type")
    def validate_registration_type(cls: Type["UserCreate"], v: str) -> str:
        if v not in ("phone", "email"):
            raise ValueError(
                "Invalid registration_type: must be either 'email' or 'phone'."
            )
        return v

    model_config = {
        "extra": "forbid",
    }


class UserUpdate(schemas.BaseUserUpdate):
    """User Update Schema"""

    name: Optional[str] = None
    phone_number: Optional[str] = None


class UserDiagnosisHistoryRequest(schemas.BaseModel):
    """User Diagnosis History Request Schema"""

    id: int
    user_id: uuid.UUID
    public_id: str
    timestamp: datetime.datetime
    uploaded_image_url: str
    result_image_url: str
    explanation_image_url: str
    gradcam_image_url: str
    report_url: str = ""
    diagnosis_data: dict

    class Config:
        from_attributes = True
