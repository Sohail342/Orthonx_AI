"""Pydantic schemas for the Contact Us feature."""

from datetime import datetime

from pydantic import BaseModel, EmailStr, field_validator


class ContactMessageCreate(BaseModel):
    """Schema for submitting a new contact message (public endpoint)."""

    name: str
    email: EmailStr
    subject: str
    message: str

    @field_validator("name", "subject", "message")
    @classmethod
    def must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be blank.")
        return v.strip()

    @field_validator("name")
    @classmethod
    def name_length(cls, v: str) -> str:
        if len(v) > 255:
            raise ValueError("Name must be 255 characters or fewer.")
        return v

    @field_validator("subject")
    @classmethod
    def subject_length(cls, v: str) -> str:
        if len(v) > 500:
            raise ValueError("Subject must be 500 characters or fewer.")
        return v


class ContactMessageRead(BaseModel):
    """Schema returned when reading a contact message (admin use)."""

    id: int
    name: str
    email: str
    subject: str
    message: str
    is_read: bool
    is_resolved: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class ContactMessageResponse(BaseModel):
    """Lightweight confirmation response after form submission."""

    message: str
