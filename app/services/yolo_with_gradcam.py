import io
import traceback
from typing import Optional
from uuid import uuid4

import cv2
import numpy as np
from fastapi import File, HTTPException, UploadFile
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.users import User
from app.repositories.user_repository import UserRepository
from app.utils.cloudinary_utils import CloudinaryUtils
from app.utils.load_yolo import load_yolo_model
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)
DETECTOR = load_yolo_model()


class YOLOGradCam:
    async def detect(
        self,
        db: AsyncSession,
        file: UploadFile = File(...),
        user: Optional[User] = None,
    ) -> dict:
        """
        Detect fractures in the uploaded image using YOLOv8
        and generate Grad-CAM visualizations.
        """
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File provided is not an image")

        detection_id = str(uuid4())

        # Initialize variables
        explanation_img = None
        gradcam_img = None
        explanation_url = ""
        gradcam_url = ""

        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            logger.info(f"Processing image: size={image.size}, mode={image.mode}")

            # Upload original uploaded image to Cloudinary
            uploaded_url, uploaded_public_id = (
                CloudinaryUtils.upload_bytes_to_cloudinary(
                    contents, public_id=f"detections/{detection_id}_uploaded"
                )
            )

            # YOLO inference using the new service
            result = DETECTOR.predict_bytes(contents)
            detections = result["detections"]
            annotated_img = result["annotated_image"]

            logger.info(f"Inference complete: {len(detections)} detections found")

            # Convert PIL annotated image to bytes for Cloudinary
            img_byte_arr = io.BytesIO()
            annotated_img.save(img_byte_arr, format="JPEG")
            result_bytes = img_byte_arr.getvalue()

            result_url, _ = CloudinaryUtils.upload_bytes_to_cloudinary(
                result_bytes, public_id=f"detections/{detection_id}_result"
            )

            # Extract detection boxes for Grad-CAM and Explanation
            detection_boxes = [d["bbox"] for d in detections]

            # Generate explanation image
            if detection_boxes:
                explanation_img = image_bgr.copy()
                for box in detection_boxes:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cv2.rectangle(explanation_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    highlight = np.zeros_like(explanation_img, dtype=np.uint8)
                    pad = 10
                    cv2.rectangle(
                        highlight,
                        (max(0, x1 - pad), max(0, y1 - pad)),
                        (
                            min(explanation_img.shape[1], x2 + pad),
                            min(explanation_img.shape[0], y2 + pad),
                        ),
                        (0, 0, 255),
                        -1,
                    )
                    explanation_img = cv2.addWeighted(
                        explanation_img, 1, highlight, 0.3, 0
                    )

                _, expl_encoded = cv2.imencode(".jpg", explanation_img)
                expl_bytes = expl_encoded.tobytes()
                explanation_url, _ = CloudinaryUtils.upload_bytes_to_cloudinary(
                    expl_bytes, public_id=f"detections/{detection_id}_explanation"
                )

                # Generate Grad-CAM image
                gradcam_img = await self.generate_gradcam(image_bgr, detection_boxes)
                _, grad_encoded = cv2.imencode(".jpg", gradcam_img)
                grad_bytes = grad_encoded.tobytes()
                gradcam_url, _ = CloudinaryUtils.upload_bytes_to_cloudinary(
                    grad_bytes, public_id=f"detections/{detection_id}_gradcam"
                )

            # Save record to database if user is provided
            if user is not None:
                repo = UserRepository(db)
                await repo.create_diagnosis_record(
                    user_id=user.id,
                    diagnosis_data={"detections": detections},
                    public_id=str(uploaded_public_id),
                    uploaded_image_url=str(uploaded_url),
                    result_image_url=str(result_url),
                    explanation_image_url=explanation_url,
                    gradcam_image_url=gradcam_url,
                    report_url="",
                )

            return {
                "detection_id": detection_id,
                "uploaded_image": str(uploaded_url),
                "result_image": str(result_url),
                "explanation_image": explanation_url,
                "gradcam_image": gradcam_url,
                "detections": detections,
            }

        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error during detection: {str(e)}"
            )

    async def generate_gradcam(self, image: np.ndarray, boxes: list) -> np.ndarray:
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


# Instantiate for API
yolo_grad_cam = YOLOGradCam()
