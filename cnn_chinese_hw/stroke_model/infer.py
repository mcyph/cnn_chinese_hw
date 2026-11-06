"""Serve generated kai outlines for characters only Tomoe has.

``generated_strokes(tomoe_strokes)`` returns a Make Me a Hanzi-format display
payload (``source: 'generated'``) or None when no checkpoint is installed. The
stroke order and count are Tomoe's; the model only supplies shape. Inference
runs on CPU (a few milliseconds per glyph) behind a lock, loading the
checkpoint on first use.
"""
import threading
from pathlib import Path

import numpy as np

from cnn_chinese_hw.get_package_dir import get_package_dir
from . import representation as R

CHECKPOINT = f'{get_package_dir()}/data/stroke_model/stroke_outline.pt'

_LOCK = threading.Lock()
_LOADED = {}
_CACHE = {}
_CACHE_MAX = 2048


def _model(path):
    with _LOCK:
        if path in _LOADED:
            return _LOADED[path]
        model = None
        if Path(path).is_file():
            import torch
            from .model import StrokeOutlineModel

            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if checkpoint.get('n') == R.N:
                model = StrokeOutlineModel()
                model.load_state_dict(checkpoint['state_dict'])
                model.eval()
        _LOADED[path] = model
        return model


def predict(tomoe_strokes, path):
    """(centre, left, right) per stroke in the 1024-unit screen frame, or None."""
    from .dataset import tomoe_input
    from .model import MAX_STROKES

    model = _model(str(path))
    if model is None or not tomoe_strokes or len(tomoe_strokes) > MAX_STROKES:
        return None
    import torch

    inputs = torch.from_numpy(np.stack([tomoe_input(s) for s in tomoe_strokes]).astype(np.float32))[None]
    mask = torch.ones(1, len(tomoe_strokes), dtype=torch.bool)
    with _LOCK, torch.inference_mode():
        centre, left, right = model(inputs, mask)
    return centre[0].numpy() * 1024, left[0].numpy() * 1024, right[0].numpy() * 1024


def generated_strokes(tomoe_strokes, variant, path, cache_key=None):
    """Display payload for ``tomoe_strokes``; cached per ``cache_key`` (e.g. the character)."""
    key = (cache_key, variant, str(path)) if cache_key is not None else None
    if key is not None and key in _CACHE:
        return _CACHE[key]
    result = _generate(tomoe_strokes, variant, path)
    if key is not None:
        with _LOCK:
            if len(_CACHE) >= _CACHE_MAX:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[key] = result
    return result


def _generate(tomoe_strokes, variant, path):
    predicted = predict(tomoe_strokes, path)
    if predicted is None:
        return None
    strokes = [{'outline': R.outline_path(c, l, r), 'points': [[round(x), round(y)] for x, y in R.to_font(c)]}
               for c, l, r in zip(*predicted)]
    return {'source': 'generated', 'variant': variant,
            'viewBox': [-40, -124, 1104, 1104], 'transform': 'translate(0 900) scale(1 -1)',
            'strokes': strokes}
