"""Locale-selected stroke geometry for teaching, separate from recognition.

Each locale prefers a source that follows its own standard:

- ``zh``: Make Me a Hanzi (PRC order). Where it disagreed with Tomoe zh,
  Tomoe was the one off the standard (何 九 为 光), and Unihan sided with
  Make Me a Hanzi on every stroke-count disagreement, so it is not screened.
- ``zh-TW``: AnimCJK's Taiwan glyphs, then Make Me a Hanzi only when it fits
  Tomoe zh-TW, then Tomoe.
- ``ja``: AnimCJK's Japanese glyphs where they agree with KanjiVG, then
  KanjiVG, then Make Me a Hanzi when it fits Tomoe ja, then Tomoe.

Where only Tomoe remains (mostly rare Traditional characters), the stroke model
(``stroke_model``) turns Tomoe's strokes into kai outlines, keeping Tomoe's
stroke order and count; without its checkpoint, Tomoe's own strokes are served.

An outline is always kept with its own medians and never mixed with another
source's strokes. The outline corpora are optional in deployments.
"""
from functools import lru_cache
import json
import math
from pathlib import Path

from .iter_makemeahanzi_data import DEFAULT_PATH
from .animcjk_data import HANT_PATH as ANIMCJK_HANT_PATH, JA_PATH as ANIMCJK_JA_PATH
from .kanjivg_strokes import DEFAULT_PATH as KANJIVG_PATH, kanjivg_strokes
from ..stroke_model.infer import CHECKPOINT as STROKE_MODEL_PATH, generated_strokes


@lru_cache(maxsize=8)
def _graphics_index(path):
    """Index byte offsets once; avoid loading the 30 MB corpus on each lookup."""
    offsets = {}
    try:
        with open(path, 'rb') as source:
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                record = json.loads(line)
                char = record.get('character')
                if isinstance(char, str) and len(char) == 1:
                    offsets[char] = offset
    except (OSError, ValueError):
        return {}
    return offsets


def _graphics(char, path, source='makemeahanzi'):
    # An optional corpus may be provisioned after the service has started.
    if not path.is_file():
        return None
    offset = _graphics_index(str(path)).get(char)
    if offset is None:
        return None
    try:
        with open(path, 'rb') as corpus:
            corpus.seek(offset)
            record = json.loads(corpus.readline())
    except (FileNotFoundError, OSError, ValueError):
        return None
    outlines, medians = record.get('strokes'), record.get('medians')
    if not isinstance(outlines, list) or not isinstance(medians, list) or not outlines or len(outlines) != len(medians):
        return None
    if not all(isinstance(d, str) and d and isinstance(points, list) and points
               and all(isinstance(p, list) and len(p) == 2 and all(isinstance(n, (int, float)) for n in p) for p in points)
               for d, points in zip(outlines, medians)):
        return None
    return {'source': source, 'variant': 'zh',
            'viewBox': [-40, -124, 1104, 1104], 'transform': 'translate(0 900) scale(1 -1)',
            'strokes': [{'outline': d, 'points': p} for d, p in zip(outlines, medians)]}


def _normalise(strokes):
    points = [point for stroke in strokes for point in stroke]
    xs, ys = zip(*points)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1)
    return [[((x - cx) / scale, (y - cy) / scale) for x, y in stroke] for stroke in strokes]


def _sample(stroke):
    if len(stroke) == 1:
        return stroke * 7
    lengths = [math.dist(a, b) for a, b in zip(stroke, stroke[1:])]
    total = sum(lengths)
    if not total:
        return [stroke[0]] * 7
    samples = []
    for i in range(7):
        distance = total * i / 6
        for j, length in enumerate(lengths):
            if distance <= length or j == len(lengths) - 1:
                t = min(1, distance / length) if length else 0
                a, b = stroke[j], stroke[j + 1]
                samples.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
                break
            distance -= length
    return samples


def _compatible(rich, strokes):
    """Conservative geometric screen for source/locale disagreements.

    Count alone misses different stroke orders. Compare direction and relative
    placement of every sampled stroke, after mapping both whole glyphs to the
    same square. This is a screen, not proof of a regional standard.
    """
    if len(rich['strokes']) != len(strokes):
        return False
    old = [stroke[4] for stroke in strokes]
    if any(not points for points in old):
        return False
    original = _normalise(old)
    candidate = _normalise([[(x, 900 - y) for x, y in stroke['points']]
                            for stroke in rich['strokes']])
    for left, right in zip(original, candidate):
        ax, ay = left[-1][0] - left[0][0], left[-1][1] - left[0][1]
        bx, by = right[-1][0] - right[0][0], right[-1][1] - right[0][1]
        length_a, length_b = math.hypot(ax, ay), math.hypot(bx, by)
        # A small reversed stroke can otherwise disappear in a glyph-wide
        # distance average. Closed strokes do not have a useful end vector.
        if length_a > 0.015 and length_b > 0.015 and (ax * bx + ay * by) / (length_a * length_b) < -0.25:
            return False
    errors = [sum(math.dist(a, b) for a, b in zip(_sample(left), _sample(right))) / 7
              for left, right in zip(original, candidate)]
    return max(errors) <= 0.32 and sum(errors) / len(errors) <= 0.14


def _agrees_with_kanjivg(rich, kanjivg):
    """Same stroke count and no stroke drawn the other way round.

    Both sources keep true proportions, so they are compared after the same
    aspect-preserving normalisation. Order is not compared: similar strokes
    make a nearest-stroke matcher unreliable, and the count and direction
    checks already catch the disagreements seen in practice.
    """
    if len(rich['strokes']) != len(kanjivg['strokes']):
        return False
    left = _normalise([[(x, 900 - y) for x, y in s['points']] for s in rich['strokes']])
    right = _normalise([s['points'] for s in kanjivg['strokes']])
    for a, b in zip(left, right):
        ax, ay = a[-1][0] - a[0][0], a[-1][1] - a[0][1]
        bx, by = b[-1][0] - b[0][0], b[-1][1] - b[0][1]
        la, lb = math.hypot(ax, ay), math.hypot(bx, by)
        if la > 0.05 and lb > 0.05 and (ax * bx + ay * by) / (la * lb) < -0.25:
            return False
    return True


def _labelled(rich, variant):
    rich['variant'] = variant
    return rich


def display_strokes(char, variant, stroke_data, graphics_path=DEFAULT_PATH,
                    kanjivg_path=KANJIVG_PATH, animcjk_ja_path=ANIMCJK_JA_PATH,
                    animcjk_hant_path=ANIMCJK_HANT_PATH, stroke_model_path=STROKE_MODEL_PATH):
    """Return one source's ordered strokes, or None when this variant is absent.

    See the module docstring for each locale's order of preference. Tomoe is
    the last resort; a locale never falls back to another locale's Tomoe data.
    """
    if variant not in ('ja', 'zh', 'zh-TW'):
        return None
    if variant == 'ja':
        japanese = kanjivg_strokes(char, kanjivg_path)
        regional = _graphics(char, Path(animcjk_ja_path), 'animcjk')
        if regional and (japanese is None or _agrees_with_kanjivg(regional, japanese)):
            return _labelled(regional, 'ja')
        if japanese:
            return japanese
    elif variant == 'zh-TW':
        regional = _graphics(char, Path(animcjk_hant_path), 'animcjk')
        if regional:
            return _labelled(regional, 'zh-TW')
    else:
        rich = _graphics(char, Path(graphics_path))
        if rich:
            return rich
    candidates = stroke_data.get(ord_=ord(char), keys_list=[variant])
    if not candidates:
        return None
    strokes = candidates[0]
    if variant != 'zh':
        rich = _graphics(char, Path(graphics_path))
        if rich and _compatible(rich, strokes):
            return _labelled(rich, variant)
    points = [stroke[4] for stroke in strokes]
    if not points or any(not p for p in points):
        return None
    generated = generated_strokes(points, variant, stroke_model_path, cache_key=char)
    if generated:
        return generated
    return {'source': 'tomoe', 'variant': variant, 'viewBox': [-40, -40, 1080, 1080],
            'strokes': [{'points': p} for p in points]}
