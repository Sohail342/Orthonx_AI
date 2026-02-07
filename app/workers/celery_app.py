"""Celery application configuration."""

from celery import Celery
from celery.signals import task_failure, task_prerun, task_success

from app.core.config import settings
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

celery_app = Celery(
    "fyp_backend",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.workers.tasks", "app.workers.yolo_tasks"],
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    broker_connection_retry_on_startup=True,
)

# Configure task routes
celery_app.conf.task_routes = {
    "app.workers.tasks.send_verification_request": {"queue": "default"},
    "app.workers.tasks.send_password_reset_email": {"queue": "default"},
    "app.workers.yolo_tasks.detect_task": {"queue": "default"},
}


@task_prerun.connect
def task_prerun_handler(task_id, task, *args, **kwargs):
    """Log when a task starts running."""
    logger.info(
        f"Task {task.name} [{task_id}] starting with args: {args}, kwargs: {kwargs}"
    )


@task_success.connect
def task_success_handler(sender, result, **kwargs):
    """Log when a task succeeds."""
    logger.info(f"Task {sender.name} succeeded with result: {result}")


@task_failure.connect
def task_failure_handler(sender, task_id, exception, traceback, *args, **kwargs):
    """Log when a task fails."""
    logger.error(f"Task {sender.name} [{task_id}] failed with exception: {exception}")
    logger.error(f"Traceback: {traceback}")
