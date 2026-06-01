"""
TensorFlow/Keras Braille cell classifier (same 26-class task as PyTorch CNN).
"""
import os
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATHS = [
    os.path.join(_REPO_ROOT, "synthetic_dataset/models/braille_cnn_tf.keras"),
    "synthetic_dataset/models/braille_cnn_tf.keras",
]
_LABELS_PATH = os.path.join(_REPO_ROOT, "synthetic_dataset/models/tf_class_labels.txt")

_model = None
_labels: Optional[List[str]] = None
_model_failed = False


def _load_labels() -> List[str]:
    global _labels
    if _labels is not None:
        return _labels
    if os.path.isfile(_LABELS_PATH):
        with open(_LABELS_PATH) as f:
            _labels = [ln.strip() for ln in f if ln.strip()]
    else:
        _labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return _labels


def _load_model():
    global _model, _model_failed
    if _model_failed:
        return None
    if _model is not None:
        return _model
    path = next((p for p in _MODEL_PATHS if os.path.isfile(p)), None)
    if not path:
        _model_failed = True
        return None
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        _model = tf.keras.models.load_model(path)
        print(f"✅ TensorFlow CNN loaded from {path}")
        return _model
    except Exception as e:
        _model_failed = True
        print(f"⚠️  TensorFlow model load failed: {e}")
        return None


def is_tensorflow_available() -> bool:
    return _load_model() is not None


class TFPredictor:
    """Mirror of CNNPredictor API for ensemble use."""

    def __init__(self):
        self.model = _load_model()
        self.labels = _load_labels()

    def predict_cell(self, cell_image: Image.Image) -> Tuple[str, float]:
        if self.model is None:
            return "?", 0.0
        try:
            img = cell_image.convert("L").resize((64, 64))
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            arr = arr.reshape(1, 64, 64, 1)
            probs = self.model.predict(arr, verbose=0)[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx]) * 100.0
            letter = self.labels[idx] if idx < len(self.labels) else "?"
            return letter, conf
        except Exception as e:
            print(f"TF predict error: {e}")
            return "?", 0.0


def get_tf_predictor() -> Optional[TFPredictor]:
    p = TFPredictor()
    return p if p.model is not None else None
