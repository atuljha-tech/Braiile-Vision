"""
YOLOv8 Braille dot detector (Ultralytics).
Trained on synthetic dot bounding boxes — used alongside OpenCV blob detection.
"""
import os
from typing import List, Tuple, Optional

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATHS = [
    os.path.join(_REPO_ROOT, "synthetic_dataset/models/braille_dots_yolov8n.pt"),
    "synthetic_dataset/models/braille_dots_yolov8n.pt",
]

_model = None
_model_failed = False


def _load_model():
    global _model, _model_failed
    if _model_failed:
        return None
    if _model is not None:
        return _model
    path = next((p for p in _MODEL_PATHS if os.path.isfile(p)), None)
    try:
        from ultralytics import YOLO
        if path:
            _model = YOLO(path)
            print(f"✅ YOLOv8 dot detector loaded from {path}")
        else:
            # Fallback: base YOLOv8n until custom weights are trained
            _model = YOLO("yolov8n.pt")
            print("✅ YOLOv8n loaded (run train_yolo.py for Braille-specific weights)")
        return _model
    except Exception as e:
        _model_failed = True
        print(f"⚠️  YOLO load failed: {e}")
        return None


def is_yolo_available() -> bool:
    return _load_model() is not None


def detect_dots(gray: np.ndarray, conf: float = 0.25) -> List[Tuple[int, int, int]]:
    """
    Run YOLO on grayscale/BGR image; return list of (cx, cy, radius).
    """
    model = _load_model()
    if model is None or gray is None or gray.size == 0:
        return []

    if len(gray.shape) == 2:
        bgr = np.stack([gray, gray, gray], axis=-1)
    else:
        bgr = gray

    try:
        results = model.predict(bgr, conf=conf, verbose=False)
    except Exception as e:
        print(f"⚠️  YOLO predict failed: {e}")
        return []

    dots = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            radius = int(max((x2 - x1), (y2 - y1)) / 2)
            radius = max(radius, 3)
            dots.append((cx, cy, radius))
    return dots
