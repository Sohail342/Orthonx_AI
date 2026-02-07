import io
from typing import Optional
from uuid import UUID

import cv2
import numpy as np
from PIL import Image

from app.models.users import DiagnosisRecord
from app.utils.cloudinary_utils import CloudinaryUtils
from app.utils.load_yolo import load_yolo_model
from app.utils.logging_utils import get_logger
from app.workers.celery_app import celery_app

logger = get_logger(__name__)
MODEL = load_yolo_model()


def create_diagnosis_record(
    db_session,
    user_id: UUID,
    diagnosis_data: Optional[dict] = None,
    public_id: Optional[str] = None,
    uploaded_image_url: Optional[str] = None,
    result_image_url: Optional[str] = None,
    explanation_image_url: Optional[str] = None,
    gradcam_image_url: Optional[str] = None,
    report_url: Optional[str] = None,
) -> DiagnosisRecord:
    new_record = DiagnosisRecord(
        user_id=user_id,
        diagnosis_data=diagnosis_data,
        public_id=public_id,
        uploaded_image_url=uploaded_image_url,
        result_image_url=result_image_url,
        explanation_image_url=explanation_image_url,
        gradcam_image_url=gradcam_image_url,
        report_url=report_url,
    )
    db_session.add(new_record)
    db_session.commit()
    db_session.refresh(new_record)
    return new_record


def generate_gradcam(image: np.ndarray, boxes: list) -> np.ndarray:
    """Generate Grad-CAM visualization"""
    vis_img = image.copy()
    if len(vis_img.shape) == 2:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)

    height, width = vis_img.shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)

    for box in boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        box_width, box_height = x2 - x1, y2 - y1
        sigma_x, sigma_y = max(box_width / 6, 10), max(box_height / 6, 10)

        y, x = np.ogrid[:height, :width]
        gaussian = np.exp(
            -(
                ((x - center_x) ** 2) / (2 * sigma_x**2)
                + ((y - center_y) ** 2) / (2 * sigma_y**2)
            )
        )
        heatmap = np.maximum(heatmap, gaussian)

    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    gradcam_visualization = cv2.addWeighted(vis_img, 0.6, heatmap_colored, 0.4, 0)

    for box in boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(gradcam_visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return gradcam_visualization


@celery_app.task(
    name="app.workers.yolo_tasks.detect_task",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=30,
    retry_kwargs={"max_retries": 3},
)
def detect_task(
    self,
    image_bytes: bytes,
    user_id: int | None,
    detection_id: str,
):
    from app.database.session import sync_SessionLocal

    db = sync_SessionLocal()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_rgb = np.array(image)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Upload original
        uploaded_url, uploaded_public_id = CloudinaryUtils.upload_bytes_to_cloudinary(
            image_bytes, public_id=f"detections/{detection_id}_uploaded"
        )

        # YOLO inference
        results = MODEL(image, conf=0.01, imgsz=1024)

        # Plot result
        results_plotted = np.ascontiguousarray(
            np.array(results[0].plot()), dtype=np.uint8
        )
        _, result_encoded = cv2.imencode(".jpg", results_plotted)

        result_url, _ = CloudinaryUtils.upload_bytes_to_cloudinary(
            result_encoded.tobytes(),
            public_id=f"detections/{detection_id}_result",
        )

        detections = []
        detection_boxes = []

        for result in results:
            for i, box in enumerate(result.boxes.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])

                detection_boxes.append([x1, y1, x2, y2])
                detections.append(
                    {
                        "id": i,
                        "class": result.names[class_id],
                        "confidence": round(float(box.conf[0]), 2),
                        "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    }
                )

        explanation_url = ""
        gradcam_url = ""

        if detection_boxes:
            # Explanation image
            explanation_img = image_bgr.copy()
            for x1, y1, x2, y2 in detection_boxes:
                cv2.rectangle(explanation_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            _, expl_encoded = cv2.imencode(".jpg", explanation_img)
            explanation_url, _ = CloudinaryUtils.upload_bytes_to_cloudinary(
                expl_encoded.tobytes(),
                public_id=f"detections/{detection_id}_explanation",
            )

            # Grad-CAM
            gradcam_img = generate_gradcam(image_bgr, detection_boxes)
            _, grad_encoded = cv2.imencode(".jpg", gradcam_img)
            gradcam_url, _ = CloudinaryUtils.upload_bytes_to_cloudinary(
                grad_encoded.tobytes(),
                public_id=f"detections/{detection_id}_gradcam",
            )

        if user_id:
            create_diagnosis_record(
                db,
                user_id=user_id,
                diagnosis_data={"detections": detections},
                public_id=str(uploaded_public_id),
                uploaded_image_url=str(uploaded_url),
                result_image_url=str(result_url),
                explanation_image_url=explanation_url,
                gradcam_image_url=gradcam_url,
                report_url="",
            )
            db.commit()

        return {
            "detection_id": detection_id,
            "uploaded_image": uploaded_url,
            "result_image": result_url,
            "explanation_image": explanation_url,
            "gradcam_image": gradcam_url,
            "detections": detections,
        }

    except Exception:
        db.rollback()
        logger.exception("Detection task failed")
        raise

    finally:
        db.close()
