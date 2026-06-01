"""
Ensemble cell classification: PyTorch CNN + TensorFlow CNN.
"""
from typing import Optional, Tuple

from PIL import Image

_pytorch = None
_tensorflow = None


def _get_pytorch():
    global _pytorch
    if _pytorch is False:
        return None
    if _pytorch is None:
        try:
            from braille_ai.cnn_predictor import CNNPredictor
            _pytorch = CNNPredictor()
            if _pytorch.model is None:
                _pytorch = False
        except Exception:
            _pytorch = False
    return _pytorch if _pytorch else None


def _get_tensorflow():
    global _tensorflow
    if _tensorflow is False:
        return None
    if _tensorflow is None:
        from braille_ai.tf_predictor import get_tf_predictor
        _tensorflow = get_tf_predictor() or False
    return _tensorflow if _tensorflow else None


def predict_cell_ensemble(cell_image: Image.Image) -> Tuple[str, float, str]:
    """
    Returns (letter, confidence_percent, backend) where backend is
    'pytorch+tensorflow', 'pytorch', 'tensorflow', or 'none'.
    """
    votes = []

    pt = _get_pytorch()
    if pt and pt.model is not None:
        letter, conf = pt.predict_cell(cell_image)
        if letter != "?":
            votes.append((letter.lower(), conf, "pytorch"))

    tf = _get_tensorflow()
    if tf and tf.model is not None:
        letter, conf = tf.predict_cell(cell_image)
        if letter != "?":
            votes.append((letter.lower(), conf, "tensorflow"))

    if not votes:
        return "?", 0.0, "none"

    if len(votes) == 1:
        return votes[0][0], votes[0][1], votes[0][2]

    # Both agree → boost confidence; else take higher confidence
    if votes[0][0] == votes[1][0]:
        conf = min(100.0, (votes[0][1] + votes[1][1]) / 2 + 10)
        return votes[0][0], conf, "pytorch+tensorflow"

    best = max(votes, key=lambda v: v[1])
    return best[0], best[1], best[2]
