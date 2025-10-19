import tensorflow as tf
from tensorflow.keras.models import Model

BODY_PARTS = [
    "XR_ELBOW",
    "XR_FINGER",
    "XR_FOREARM",
    "XR_HAND",
    "XR_HUMERUS",
    "XR_SHOULDER",
    "XR_WRIST",
]


def model_load(model_path: str) -> Model:
    """Load a Keras model from the given file path."""
    model = tf.keras.models.load_model(model_path)
    return model
