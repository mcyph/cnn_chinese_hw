"""Synthetic Tomoe-style inputs from outline medians.

Real Tomoe input differs from an outline font's medians in measurable ways
(training pairs, per-axis-normalised glyph box; see the README):

- every glyph is stretched to fill the box on both axes;
- strokes are sparse polylines, mostly 2-4 vertices (medians have 4-9);
- each stroke is shifted as a whole (median 0.020 of the box), wobbles
  within itself (median 0.011) and runs a little longer at its ends
  (endpoint gap median 0.030).

``tomoe_like`` reproduces those so every outline glyph, not only the ~7,700
with a matching Tomoe entry, can train the model. The parameters below are set
from those measurements; ``python -m cnn_chinese_hw.stroke_model.synthetic``
re-measures the generator against the real pairs.
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from cnn_chinese_hw.get_package_dir import get_package_dir
from cnn_chinese_hw.parse_data.animcjk_data import HANT_PATH, JA_PATH
from cnn_chinese_hw.parse_data.iter_makemeahanzi_data import DEFAULT_PATH as MMH_PATH
from . import dataset as D
from . import representation as R

CACHE = Path(f'{get_package_dir()}/data/stroke_model/synthetic.npz')
SHIFT_SIGMA = 0.011       # per-stroke drift (extension and simplification add the rest of the measured 0.020)
WOBBLE_SIGMA = 0.009      # per-vertex noise; median 0.011
EXTEND = (0.0, 0.04)      # extra length at each end, fraction of the box
SIMPLIFY = (0.02, 0.06)   # Douglas-Peucker tolerance, fraction of the box


def _rdp(points, tolerance):
    """Ramer-Douglas-Peucker simplification, keeping both ends."""
    if len(points) < 3:
        return points
    a, b = points[0], points[-1]
    ab = b - a
    norm = np.linalg.norm(ab)
    rel = points[1:-1] - a
    dist = (np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / norm) if norm else np.linalg.norm(rel, axis=1)
    i = int(np.argmax(dist))
    if dist[i] <= tolerance:
        return np.vstack([a, b])
    left = _rdp(points[:i + 2], tolerance)
    right = _rdp(points[i + 1:], tolerance)
    return np.vstack([left[:-1], right])


def tomoe_like(medians_screen, rng):
    """Tomoe-frame strokes (0..1000, y down) from screen-coordinate medians."""
    strokes = [np.asarray(m, dtype=float) for m in medians_screen]
    pts = np.concatenate(strokes)
    lo, hi = pts.min(0), pts.max(0)
    span = np.where(hi - lo < 1e-6, 1, hi - lo)
    out = []
    for s in strokes:
        s = (s - lo) / span
        if len(s) == 1:
            s = np.vstack([s, s + 1e-3])
        for end, inward in ((0, 1), (-1, -2)):
            v = s[end] - s[inward]
            n = np.linalg.norm(v)
            if n > 1e-6:
                s[end] = s[end] + v / n * rng.uniform(*EXTEND)
        s = _rdp(s, rng.uniform(*SIMPLIFY))
        s = s + rng.normal(0, SHIFT_SIGMA, 2) + rng.normal(0, WOBBLE_SIGMA, s.shape)
        out.append(s)
    # Tomoe fills its box exactly on both axes.
    pts = np.concatenate(out)
    lo, hi = pts.min(0), pts.max(0)
    span = np.where(hi - lo < 1e-6, 1, hi - lo)
    return [(s - lo) / span * 1000 for s in out]


def _encode(job):
    char, source, rec = job
    try:
        enc = [R.encode(d, m) for d, m in zip(rec['strokes'], rec['medians'])]
    except Exception:
        return None
    medians = [np.array([(x, 900 - y) for x, y in m], dtype=np.float32) for m in rec['medians']]
    return (char, source, medians,
            np.stack([e[0] for e in enc]).astype(np.float32) / 1024,
            np.stack([e[1] for e in enc]).astype(np.float32) / 1024,
            np.stack([e[2] for e in enc]).astype(np.float32) / 1024)


def build(workers=12, cache=CACHE):
    """Encode every train-split outline glyph (all three sources) with its medians."""
    jobs = []
    for path in (MMH_PATH, JA_PATH, HANT_PATH):
        if not Path(path).is_file():
            continue
        for line in open(path, encoding='utf-8'):
            rec = json.loads(line)
            char = rec.get('character')
            if (isinstance(char, str) and len(char) == 1 and D.split_of(char) == 'train'
                    and rec.get('medians') and len(rec['medians']) == len(rec.get('strokes', []))
                    and len(rec['medians']) <= D.MAX_STROKES):
                jobs.append((char, Path(path).name, rec))
    with ProcessPoolExecutor(workers) as pool:
        rows = [r for r in pool.map(_encode, jobs, chunksize=64) if r]
    stroke_counts = np.array([len(r[2]) for r in rows])
    medians = [m for r in rows for m in r[2]]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, chars=np.array([r[0] for r in rows]), sources=np.array([r[1] for r in rows]),
        counts=stroke_counts, median_lengths=np.array([len(m) for m in medians]),
        medians=np.concatenate(medians), centre=np.concatenate([r[3] for r in rows]),
        left=np.concatenate([r[4] for r in rows]), right=np.concatenate([r[5] for r in rows]))
    return len(jobs), len(rows)


def load(cache=CACHE):
    with np.load(cache) as archive:
        z = {key: archive[key] for key in archive.files}
    offsets = np.concatenate([[0], np.cumsum(z['counts'])])
    m_offsets = np.concatenate([[0], np.cumsum(z['median_lengths'])])
    glyphs = []
    for i, ch in enumerate(z['chars']):
        a, b = offsets[i], offsets[i + 1]
        glyphs.append({'char': str(ch), 'source': str(z['sources'][i]),
                       'medians': [z['medians'][m_offsets[k]:m_offsets[k + 1]] for k in range(a, b)],
                       'centre': z['centre'][a:b], 'left': z['left'][a:b], 'right': z['right'][a:b]})
    return glyphs


def sample(glyph, rng):
    """One fresh training example (same dict shape as ``dataset.load``)."""
    strokes = tomoe_like(glyph['medians'], rng)
    return {'inputs': np.stack([D.tomoe_input(s) for s in strokes]).astype(np.float32),
            'centre': glyph['centre'], 'left': glyph['left'], 'right': glyph['right']}


def measure(pairs, sources, rng, limit=3000):
    """Generator statistics on the real pairs' own medians, next to real Tomoe's."""
    real, synth = {'vertices': [], 'shift': [], 'wobble': [], 'endgap': []}, None
    synth = {k: [] for k in real}

    def stats(strokes, medians, into):
        a, b = D._per_axis(strokes), D._per_axis(medians)
        ts = np.linspace(0, 1, 16)
        for x, y in zip(a, b):
            d = R.resample(x, ts) - R.resample(y, ts)
            into['vertices'].append(len(x))
            into['shift'].append(np.linalg.norm(d.mean(0)))
            into['wobble'].append(np.linalg.norm(d - d.mean(0), axis=1).mean())
            into['endgap'].append((np.linalg.norm(x[0] - y[0]) + np.linalg.norm(x[-1] - y[-1])) / 2)

    for char, locale, target, tomoe in pairs[:limit]:
        rec = sources[target][char]
        medians = [np.array([(x, 900 - y) for x, y in m], float) for m in rec['medians']]
        stats([np.asarray(t, float) for t in tomoe], medians, real)
        stats(tomoe_like(medians, rng), medians, synth)
    q = lambda v: np.percentile(v, [25, 50, 75, 90]).round(3).tolist()  # noqa: E731
    for k in real:
        print(f'{k:9s} real {q(real[k])}   synthetic {q(synth[k])}')


if __name__ == '__main__':
    import sys
    import time
    from cnn_chinese_hw.parse_data.StrokeData import StrokeData

    if '--measure' in sys.argv:
        sd = StrokeData()
        D._init()
        by_name = {Path(p).name: D._SOURCES[p] for paths in D._TARGETS.values() for p in paths}
        pairs = [(g['char'], g['locale'], g['target'],
                  [s[4] for s in sd.D[g['locale']][1][ord(g['char'])]])
                 for g in D.load() if g['split'] == 'train']
        measure(pairs, by_name, np.random.default_rng(0))
    else:
        t = time.time()
        total, kept = build()
        print(f'{kept}/{total} train-split outline glyphs encoded in {time.time() - t:.0f}s -> {CACHE}')
