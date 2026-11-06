"""Training pairs: Tomoe stroke trajectories -> kai outline representation.

A pair is a character whose Tomoe strokes (any of ja / zh / zh-TW) line up one
to one with an outline source in the same order and direction. The target is
AnimCJK's regional glyph where it has one (Taiwan for zh-TW, Japanese for ja),
else Make Me a Hanzi; all three descend from the same Arphic kai fonts, so the
target style is consistent.

Tomoe stretches every glyph to fill its square, so the matcher normalises each
axis separately on both sides; an aspect-preserving comparison would reject
every flat or narrow character (一, 日).

Arrays (float32, per stroke): ``inputs`` = Tomoe stroke resampled at the
representation's spacing in Tomoe's 0..1 frame; ``centre``/``left``/``right``
= target in the 1024-unit screen frame divided by 1024.
"""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from cnn_chinese_hw.get_package_dir import get_package_dir
from cnn_chinese_hw.parse_data.animcjk_data import HANT_PATH, JA_PATH
from cnn_chinese_hw.parse_data.iter_makemeahanzi_data import DEFAULT_PATH as MMH_PATH
from . import representation as R

CACHE = Path(f'{get_package_dir()}/data/stroke_model/pairs.npz')
MAX_STROKES = 40
_TARGETS = {'zh': (MMH_PATH,), 'zh-TW': (HANT_PATH, MMH_PATH), 'ja': (JA_PATH, MMH_PATH)}


def split_of(char):
    """Stable split by character, so every locale's copy lands together."""
    h = int(hashlib.sha1(char.encode('utf-8')).hexdigest()[:8], 16) % 100
    return 'test' if h < 5 else 'val' if h < 10 else 'train'


def tomoe_input(stroke):
    return R.resample(stroke, R.cosine_spacing()) / 1000.0


def _per_axis(strokes):
    pts = np.concatenate([np.asarray(s, dtype=float) for s in strokes])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.where(hi - lo < 1e-6, 1, hi - lo)
    return [(np.asarray(s, dtype=float) - lo) / span for s in strokes]


def matches(tomoe, medians_screen, max_error=0.12, mean_error=0.06):
    """Same count, no reversed stroke, and close placement after per-axis fits."""
    if len(tomoe) != len(medians_screen):
        return False
    ts = R.cosine_spacing(7)
    a, b = _per_axis(tomoe), _per_axis(medians_screen)
    errors = []
    for x, y in zip(a, b):
        dx, dy = x[-1] - x[0], y[-1] - y[0]
        la, lb = np.linalg.norm(dx), np.linalg.norm(dy)
        if la > 0.05 and lb > 0.05 and float(np.dot(dx, dy)) / (la * lb) < -0.25:
            return False
        errors.append(float(np.linalg.norm(R.resample(x, ts) - R.resample(y, ts), axis=1).mean()))
    return max(errors) <= max_error and sum(errors) / len(errors) <= mean_error


def _load(path):
    out = {}
    if Path(path).is_file():
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            out[r['character']] = r
    return out


_SOURCES = {}


def _encode_job(job):
    locale, char, tomoe = job
    for path in _TARGETS[locale]:
        rec = _SOURCES[path].get(char)
        if not rec or len(rec['medians']) != len(tomoe) or len(tomoe) > MAX_STROKES:
            continue
        medians = [[(x, 900 - y) for x, y in m] for m in rec['medians']]
        if not matches(tomoe, medians):
            continue
        try:
            enc = [R.encode(d, m) for d, m in zip(rec['strokes'], rec['medians'])]
        except Exception:
            return None
        return (char, locale, Path(path).name,
                np.stack([tomoe_input(s) for s in tomoe]).astype(np.float32),
                np.stack([e[0] for e in enc]).astype(np.float32) / 1024,
                np.stack([e[1] for e in enc]).astype(np.float32) / 1024,
                np.stack([e[2] for e in enc]).astype(np.float32) / 1024)
    return None


def _init():
    for paths in _TARGETS.values():
        for p in paths:
            if p not in _SOURCES:
                _SOURCES[p] = _load(p)


def build(stroke_data, workers=12, cache=CACHE):
    """Encode every matching pair and save a flat, ragged npz."""
    jobs = [(locale, chr(o), [list(map(tuple, s[4])) for s in strokes])
            for locale in ('zh', 'zh-TW', 'ja')
            for o, strokes in stroke_data.D[locale][1].items()
            if strokes and all(s[4] for s in strokes)]
    with ProcessPoolExecutor(workers, initializer=_init) as pool:
        rows = [r for r in pool.map(_encode_job, jobs, chunksize=64) if r]
    counts = np.array([len(r[3]) for r in rows])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        chars=np.array([r[0] for r in rows]), locales=np.array([r[1] for r in rows]),
        targets=np.array([r[2] for r in rows]), counts=counts,
        inputs=np.concatenate([r[3] for r in rows]), centre=np.concatenate([r[4] for r in rows]),
        left=np.concatenate([r[5] for r in rows]), right=np.concatenate([r[6] for r in rows]))
    return len(jobs), len(rows)


def load(cache=CACHE):
    """List of glyph dicts with per-stroke arrays, plus each glyph's split."""
    # Read each array once: indexing an NpzFile decompresses the whole array
    # on every access.
    with np.load(cache) as archive:
        z = {key: archive[key] for key in archive.files}
    offsets = np.concatenate([[0], np.cumsum(z['counts'])])
    glyphs = []
    for i, ch in enumerate(z['chars']):
        a, b = offsets[i], offsets[i + 1]
        glyphs.append({'char': str(ch), 'locale': str(z['locales'][i]), 'target': str(z['targets'][i]),
                       'split': split_of(str(ch)), 'inputs': z['inputs'][a:b], 'centre': z['centre'][a:b],
                       'left': z['left'][a:b], 'right': z['right'][a:b]})
    return glyphs


if __name__ == '__main__':
    import time
    from cnn_chinese_hw.parse_data.StrokeData import StrokeData

    t = time.time()
    total, kept = build(StrokeData())
    print(f'{kept}/{total} Tomoe glyphs paired in {time.time() - t:.0f}s -> {CACHE}')
