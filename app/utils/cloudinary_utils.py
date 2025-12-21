import io

import cloudinary
import cloudinary.uploader
from fastapi import HTTPException

from app.core.config import settings
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET,
)


class CloudinaryUtils:
    @staticmethod
    def upload_image(file_path: str, public_id: str) -> tuple:
        """Upload an image to Cloudinary.

        Args:
            file_path (str): The local path to the image file.
            public_id (str, optional): The public ID to assign to the uploaded image.

        Returns:
            dict: The response from Cloudinary containing details about the uploaded image.
        """
        try:
            response = cloudinary.uploader.upload(
                file_path, public_id=public_id, overwrite=True, resource_type="image"
            )
            return response["secure_url"], response["public_id"]
        except Exception as e:
            raise RuntimeError(f"Failed to upload image to Cloudinary: {e}")

    @staticmethod
    def upload_bytes_to_cloudinary(
        file_bytes: bytes, public_id: str
    ) -> tuple[str, str]:
        """Upload image bytes to Cloudinary and return secure_url"""
        try:
            file_like = io.BytesIO(file_bytes)
            result = cloudinary.uploader.upload(
                file_like, public_id=public_id, overwrite=True, resource_type="image"
            )
            return result["secure_url"], result["public_id"]
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Cloudinary upload failed: {e}"
            )

    @staticmethod
    def delete_image(public_id: str) -> dict:  # type: ignore
        """Delete an image from Cloudinary.

        Args:
            public_id (str): The public ID of the image to delete.

        Returns:
            dict: The response from Cloudinary regarding the deletion status.
        """
        try:
            response = cloudinary.uploader.destroy(public_id, resource_type="image")
            return response  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to delete image from Cloudinary: {e}")
