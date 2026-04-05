"""Celery task for generating PDF diagnosis reports in the background."""

from app.utils.logging_utils import get_logger
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(
    name="app.workers.report_tasks.generate_report_task",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=30,
    retry_kwargs={"max_retries": 3},
)
def generate_report_task(self, record_id: int) -> dict:
    """Generate a PDF report for a diagnosis record.

    This task:
    1. Fetches the DiagnosisRecord and User from the database.
    2. Calls Ollama (llama3.2:3b) to generate a clinical summary.
    3. Builds a professional PDF with ReportLab.
    4. Uploads the PDF to Cloudinary.
    5. Updates the record's report_url in the database.
    """
    import app.models.appointment  # noqa
    from app.database.session import sync_SessionLocal
    from app.models.diagnosis import DiagnosisRecord
    from app.models.users import User
    from app.utils.cloudinary_utils import CloudinaryUtils
    from app.utils.ollama_utils import generate_clinical_summary
    from app.utils.report_builder import build_diagnosis_pdf

    db = sync_SessionLocal()

    try:
        record = (
            db.query(DiagnosisRecord).filter(DiagnosisRecord.id == record_id).first()
        )
        if not record:
            logger.error(f"DiagnosisRecord {record_id} not found — skipping report")
            return {"status": "error", "message": f"Record {record_id} not found"}

        user = db.query(User).filter(User.id == record.user_id).first()
        user_name = "Unknown"
        user_email = "N/A"
        if user:
            user_name = (
                user.name or user.email.split("@")[0] if user.email else "Unknown"
            )
            user_email = user.email or "N/A"

        # Generate AI clinical summary
        detections = []
        if record.diagnosis_data and isinstance(record.diagnosis_data, dict):
            detections = record.diagnosis_data.get("detections", [])

        logger.info(
            f"Generating clinical summary for record {record_id} with {len(detections)} detections"
        )
        ai_summary = generate_clinical_summary(detections)

        # Build PDF
        record_data = {
            "id": record.id,
            "public_id": record.public_id,
            "timestamp": record.timestamp,
            "diagnosis_data": record.diagnosis_data,
            "uploaded_image_url": record.uploaded_image_url,
            "result_image_url": record.result_image_url,
            "gradcam_image_url": record.gradcam_image_url,
        }
        user_data = {
            "name": user_name,
            "email": user_email,
        }

        logger.info(f"Building PDF report for record {record_id}")
        pdf_bytes = build_diagnosis_pdf(record_data, user_data, ai_summary)

        # 5. Upload PDF to Cloudinary (use record.id for a clean path, add .pdf extension)
        pdf_public_id = f"reports/diagnosis_report_{record.id}.pdf"
        report_url, _ = CloudinaryUtils.upload_pdf_to_cloudinary(
            pdf_bytes, public_id=pdf_public_id, format="pdf"
        )

        # Update the record in DB
        record.report_url = report_url
        db.commit()

        logger.info(
            f"Report generated and uploaded for record {record_id}: {report_url}"
        )

        return {
            "status": "success",
            "record_id": record_id,
            "report_url": report_url,
        }

    except Exception:
        db.rollback()
        logger.exception(f"Report generation failed for record {record_id}")
        raise

    finally:
        db.close()
