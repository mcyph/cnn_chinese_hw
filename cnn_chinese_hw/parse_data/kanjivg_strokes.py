"""KanjiVG stroke centrelines for Japanese stroke-order display.

KanjiVG draws each stroke as a cubic Bézier centreline in Japanese standard
order, labelled with its CJK stroke type (U+31C0..U+31E3). The display wants
points in the same 1000-unit, y-down frame as Tomoe, so each curve is sampled
here; the stroke type travels as ``kind`` so the renderer can shape the stroke
ending (hook, sweep, stop) from data instead of guessing.

KanjiVG is CC BY-SA 3.0 (© Ulrich Apel / the KanjiVG project); see
THIRD_PARTY_DATA.md. The file is optional in deployments.
"""
from functools import lru_cache
import math
from pathlib import Path
import re

from svg.path import Move, parse_path

from cnn_chinese_hw.get_package_dir import get_package_dir

DEFAULT_PATH = f'{get_package_dir()}/data/validation/kanjivg.xml'
VIEWBOX = [0, 0, 1000, 1000]   # KanjiVG's own canvas, margins included
_SCALE = 1000 / 109       # KanjiVG's canvas is 109 x 109
_STEP = 2.5               # KanjiVG units between samples along a curve
_KANJI = re.compile(rb'<kanji id="kvg:kanji_([0-9a-f]+)">')
_D = re.compile(r'\sd="([^"]+)"')
_TYPE = re.compile(r'kvg:type="([^"]*)"')


@lru_cache(maxsize=2)
def _index(path):
    """Byte offset of every character's block, read once per file."""
    offsets = {}
    try:
        with open(path, 'rb') as source:
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                match = _KANJI.match(line)
                if match:
                    offsets[chr(int(match.group(1), 16))] = offset
    except (OSError, ValueError):
        return {}
    return offsets


def stroke_kind(kvg_type):
    """The leading CJK stroke character of a KanjiVG type, or None.

    Types carry a variant letter (``㇐a``) or an alternative (``㇔/㇏``); the
    first stroke character is the primary reading.
    """
    for ch in kvg_type or '':
        if '㇀' <= ch <= '㇣':
            return ch
        if ch == '/':
            break
    return None


def sample_path(d):
    """Sample an SVG path into display-frame points, keeping segment joins.

    Every segment end is a sample, so a corner between two segments stays a
    single sharp vertex that the renderer can recognise as a corner.
    """
    points = []
    for segment in parse_path(d):
        if isinstance(segment, Move):
            continue
        if not points:
            points.append(segment.start)
        count = max(1, math.ceil(segment.length(error=1e-3) / _STEP))
        points.extend(segment.point(i / count) for i in range(1, count + 1))
    out = []
    for p in points:
        xy = [round(p.real * _SCALE, 1), round(p.imag * _SCALE, 1)]
        if not out or xy != out[-1]:
            out.append(xy)
    return out


@lru_cache(maxsize=512)
def _strokes(char, path):
    offset = _index(path).get(char)
    if offset is None:
        return None
    strokes = []
    try:
        with open(path, 'rb') as source:
            source.seek(offset)
            source.readline()
            for raw in source:
                line = raw.decode('utf-8')
                if line.startswith('</kanji>'):
                    break
                d = _D.search(line) if '<path' in line else None
                if not d:
                    continue
                points = sample_path(d.group(1))
                if not points:
                    return None
                kind = stroke_kind((_TYPE.search(line) or [None, None])[1])
                strokes.append({'points': points, **({'kind': kind} if kind else {})})
    except (OSError, ValueError):
        return None
    return tuple(strokes) or None


def kanjivg_strokes(char, path=DEFAULT_PATH):
    """Ordered KanjiVG strokes for ``char`` as a display payload, or None."""
    path = Path(path)
    if not path.is_file():
        return None
    strokes = _strokes(char, str(path))
    if not strokes:
        return None
    return {'source': 'kanjivg', 'variant': 'ja', 'viewBox': list(VIEWBOX),
            'strokes': [dict(s, points=[list(p) for p in s['points']]) for s in strokes]}
