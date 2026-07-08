import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from app.core.security import get_jwt_strategy
from app.models.contact import ContactMessage
from app.models.users import SystemConfiguration, User
from app.models.verification import DoctorVerification
from app.schemas.doctor import VerificationStatus
from app.schemas.user import UserType

pytestmark = pytest.mark.asyncio


# Helper function to generate authorization headers for any user
async def get_auth_headers(user: User):
    jwt_strategy = get_jwt_strategy()
    token = await jwt_strategy.write_token(user)
    return {"Authorization": f"Bearer {token}"}


async def create_test_user(
    db_session, email: str, user_type: UserType, is_superuser: bool = False
) -> User:
    user = User(
        id=uuid.uuid4(),
        email=email,
        hashed_password="fastapiusershashedpasswordmock",
        is_active=True,
        is_superuser=is_superuser,
        is_verified=True,
        user_type=user_type,
        credits=200,
        name=f"Test {user_type.value}",
        last_credit_reset_date=datetime.now(timezone.utc),
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


async def test_root_endpoint(client):
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_contact_submit_public(client, db_session):
    payload = {
        "name": "Jane Doe",
        "email": "jane@example.com",
        "subject": "Platform feedback",
        "message": "Orthonx is doing great. Keep it up!",
    }
    response = await client.post("/api/v1/contact/", json=payload)
    assert response.status_code == 201
    assert "message" in response.json()

    # Verify message in database
    stmt = select(ContactMessage).where(ContactMessage.email == "jane@example.com")
    result = await db_session.execute(stmt)
    msg = result.scalar_one_or_none()
    assert msg is not None
    assert msg.name == "Jane Doe"
    assert msg.subject == "Platform feedback"
    assert msg.is_read is False
    assert msg.is_resolved is False


async def test_contact_admin_access_control(client, db_session):
    # Create non-admin and admin users
    patient = await create_test_user(db_session, "patient@example.com", UserType.USER)
    admin = await create_test_user(
        db_session, "admin@example.com", UserType.USER, is_superuser=True
    )

    # Insert a dummy contact message
    msg = ContactMessage(
        name="Visitor",
        email="visitor@example.com",
        subject="Help",
        message="I cannot log in.",
    )
    db_session.add(msg)
    await db_session.commit()
    await db_session.refresh(msg)

    # List messages - Non-admin should get 403
    patient_headers = await get_auth_headers(patient)
    response = await client.get("/api/v1/contact/", headers=patient_headers)
    assert response.status_code == 403

    # List messages - Admin should get 200
    admin_headers = await get_auth_headers(admin)
    response = await client.get("/api/v1/contact/", headers=admin_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert any(x["email"] == "visitor@example.com" for x in data)

    # Mark as read - Non-admin should get 403
    response = await client.patch(
        f"/api/v1/contact/{msg.id}/read", headers=patient_headers
    )
    assert response.status_code == 403

    # Mark as read - Admin should get 200
    response = await client.patch(
        f"/api/v1/contact/{msg.id}/read", headers=admin_headers
    )
    assert response.status_code == 200
    assert response.json()["is_read"] is True

    # Mark as resolved - Admin should get 200
    response = await client.patch(
        f"/api/v1/contact/{msg.id}/resolve", headers=admin_headers
    )
    assert response.status_code == 200
    updated_data = response.json()
    assert updated_data["is_resolved"] is True
    assert updated_data["is_read"] is True


async def test_appointments_list_doctors(client, db_session):
    patient = await create_test_user(
        db_session, "patient_appt@example.com", UserType.USER
    )
    patient_headers = await get_auth_headers(patient)

    # Create approved doctor
    doc1 = await create_test_user(
        db_session, "approved_doc@example.com", UserType.DOCTOR
    )
    verification1 = DoctorVerification(
        doctor_id=doc1.id,
        full_name="Dr. Approved",
        specialty="Orthopedics",
        license_number="LIC-111",
        status=VerificationStatus.APPROVED,
    )
    db_session.add(verification1)

    # Create pending doctor
    doc2 = await create_test_user(
        db_session, "pending_doc@example.com", UserType.DOCTOR
    )
    verification2 = DoctorVerification(
        doctor_id=doc2.id,
        full_name="Dr. Pending",
        specialty="General Medicine",
        license_number="LIC-222",
        status=VerificationStatus.PENDING,
    )
    db_session.add(verification2)
    await db_session.commit()

    # List doctors
    response = await client.get("/api/v1/appointments/doctors", headers=patient_headers)
    assert response.status_code == 200
    docs = response.json()

    # Assert only approved doctor is listed
    assert any(d["id"] == str(doc1.id) for d in docs)
    assert not any(d["id"] == str(doc2.id) for d in docs)


async def test_dashboard_config_and_credits(client, db_session):
    # Create superadmin and user
    admin = await create_test_user(
        db_session, "config_admin@example.com", UserType.USER, is_superuser=True
    )
    patient = await create_test_user(
        db_session, "config_patient@example.com", UserType.USER
    )

    admin_headers = await get_auth_headers(admin)
    patient_headers = await get_auth_headers(patient)

    # Seed system configs
    configs = [
        SystemConfiguration(key="DEFAULT_CREDITS", value=200),
        SystemConfiguration(key="UPLOAD_IMAGE_COST", value=10),
        SystemConfiguration(key="REQUEST_REVIEW_COST", value=20),
        SystemConfiguration(key="DOCTOR_REVIEW_EARNING", value=15),
        SystemConfiguration(key="APPOINTMENT_BOOKING_COST", value=50),
        SystemConfiguration(key="APPOINTMENT_COMPLETION_EARNING", value=40),
    ]
    for config in configs:
        db_session.add(config)
    await db_session.commit()

    # Get config (available to active authenticated users)
    response = await client.get("/api/v1/dashboard/config", headers=patient_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["default_credits"] == 200
    assert data["upload_image_cost"] == 10

    # Put config - normal user should get 403
    update_payload = {
        "default_credits": 300,
        "upload_image_cost": 15,
        "request_review_cost": 25,
        "doctor_review_earning": 20,
        "appointment_booking_cost": 60,
        "appointment_completion_earning": 45,
    }
    response = await client.put(
        "/api/v1/dashboard/config", json=update_payload, headers=patient_headers
    )
    assert response.status_code == 403

    # Put config
    response = await client.put(
        "/api/v1/dashboard/config", json=update_payload, headers=admin_headers
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Configuration updated successfully"

    # Verify updated values in db
    response = await client.get("/api/v1/dashboard/config", headers=patient_headers)
    assert response.json()["default_credits"] == 300

    # Update credits of a user
    credit_payload = {"credits": 500}
    response = await client.put(
        f"/api/v1/dashboard/users/{patient.id}/credits",
        json=credit_payload,
        headers=patient_headers,
    )
    assert response.status_code == 403

    # Update credits of a user
    response = await client.put(
        f"/api/v1/dashboard/users/{patient.id}/credits",
        json=credit_payload,
        headers=admin_headers,
    )
    assert response.status_code == 200
    assert response.json()["credits"] == 500

    # Get users list
    response = await client.get("/api/v1/dashboard/users", headers=admin_headers)
    assert response.status_code == 200
    users_data = response.json()
    assert users_data["total"] >= 2
    assert any(u["id"] == str(patient.id) for u in users_data["items"])
