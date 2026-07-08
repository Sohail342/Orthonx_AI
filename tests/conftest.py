import sys
from unittest.mock import MagicMock

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Mock BoneFractureDetector & load_yolo_model
mock_detector = MagicMock()
mock_load_yolo = MagicMock()
mock_load_yolo.load_yolo_model = MagicMock(return_value=mock_detector)
sys.modules["app.utils.load_yolo"] = mock_load_yolo

from app.database.base_class import Base  # noqa
from app.database.session import get_db  # noqa
from app.main import app  # noqa

# Use in-memory SQLite for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session", autouse=True)
async def setup_test_db():
    # Setup connection
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(setup_test_db):
    engine = setup_test_db
    connection = await engine.connect()
    transaction = await connection.begin()

    # Session constructor
    async_session = sessionmaker(
        bind=connection,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    session = async_session()

    yield session

    await session.close()
    await transaction.rollback()
    await connection.close()


@pytest.fixture
async def client(db_session):
    async def override_get_db():
        yield db_session

    from httpx import ASGITransport

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def mock_celery_tasks(monkeypatch):
    from unittest.mock import MagicMock

    mock_delay = MagicMock()
    # Mock celery tasks .delay methods to avoid contacting Redis
    try:
        from app.workers import tasks

        monkeypatch.setattr(tasks.send_verification_request, "delay", mock_delay)
        monkeypatch.setattr(tasks.send_password_reset_email, "delay", mock_delay)
    except Exception:
        pass
    return mock_delay
