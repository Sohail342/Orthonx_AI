import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


class FractureDetector:
    def __init__(self, onnx_model_path: str) -> None:
        self.session = ort.InferenceSession(onnx_model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for ONNX model"""
        image = image.resize((224, 224))
        image_array = np.array(image).astype(np.float32)

        if image_array.shape[-1] == 4:  # RGBA
            image_array = image_array[..., :3]

        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def detect_fracture_regions(
        self, image_array: np.ndarray, threshold: float = 0.3
    ) -> list:
        """Detect potential fracture regions using image processing"""
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array

        # Multiple detection methods
        regions = []

        # Method 1: Edge detection
        edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 30:  # Filter small areas
                x, y, w, h = cv2.boundingRect(contour)
                regions.append(
                    {
                        "type": "edge",
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "area": area,
                        "confidence": min(area / 1000, 1.0),  # Normalized confidence
                    }
                )

        return regions

    def predict(self, image: Image.Image, fracture_threshold: float = 0.5) -> dict:
        """Make prediction with fracture region detection"""
        # Preprocess for model
        input_array = self.preprocess_image(image)

        # Run ONNX inference
        prediction = self.session.run(
            [self.output_name], {self.input_name: input_array}
        )[0]
        prob = float(prediction[0][0])

        # Detect fracture regions on original image
        original_array = np.array(image)
        fracture_regions = self.detect_fracture_regions(original_array)

        # Filter regions based on prediction confidence
        if prob > fracture_threshold:
            filtered_regions = [
                region for region in fracture_regions if region["confidence"] > 0.1
            ]
        else:
            filtered_regions = []

        return {
            "probability": prob,
            "prediction": "fracture" if prob > fracture_threshold else "no_fracture",
            "confidence": prob if prob > fracture_threshold else 1 - prob,
            "fracture_regions": filtered_regions,
            "regions_count": len(filtered_regions),
        }
