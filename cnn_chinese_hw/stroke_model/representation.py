"""A stroke outline as a centreline with a left and right width at each sample.

This is what the stroke model learns to predict: ``N`` centreline points from
tip to tip of the outline plus the distance from each point to the outline on
either side. Samples are cosine-spaced so they crowd the two ends, where a kai
brush stroke changes shape fastest (press, hook, taper); the end samples sit
on the outline itself, so their widths are close to zero.

All coordinates here are *screen* coordinates (y down) in Make Me a Hanzi's
1024-unit frame: ``y_screen = 900 - y_font``. ``to_font`` converts back so a
decoded outline can be served exactly like a Make Me a Hanzi one.
"""
import numpy as np
from shapely.geometry import LineString, Point, Polygon  # noqa: F401
from shapely.validation import make_valid
from svg.path import Move, parse_path

N = 24
_RAY = 400.0          # longest width / end extension searched for, in em units


def cosine_spacing(n=N):
    return (1 - np.cos(np.pi * np.arange(n) / (n - 1))) / 2


def resample(points, ts):
    """Points at arc-length fractions ``ts`` (0..1) along a polyline."""
    p = np.asarray(points, dtype=float)
    if len(p) == 1:
        return np.repeat(p, len(ts), axis=0)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    if cum[-1] == 0:
        return np.repeat(p[:1], len(ts), axis=0)
    s = np.asarray(ts) * cum[-1]
    return np.stack([np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])], axis=1)


def outline_polygon(d, samples_per_segment=8):
    """An SVG outline (font coordinates) as a screen-coordinate polygon."""
    pts = []
    for segment in parse_path(d):
        if isinstance(segment, Move):
            continue
        for i in range(samples_per_segment):
            z = segment.point(i / samples_per_segment)
            pts.append((z.real, 900 - z.imag))
    poly = Polygon(pts)
    return poly if poly.is_valid else make_valid(poly)


def _tangents(c):
    t = np.gradient(c, axis=0)
    norm = np.linalg.norm(t, axis=1, keepdims=True)
    return t / np.where(norm == 0, 1, norm)


def _hit(poly, origin, direction, inside):
    """Distance from ``origin`` to the outline along ``direction``.

    From inside, the nearest boundary crossing ahead; from outside (a median
    that strays past the outline), the nearest crossing behind, as a negative
    distance.
    """
    o = np.asarray(origin)
    ray = LineString([tuple(o - direction * _RAY), tuple(o + direction * _RAY)])
    hits = ray.intersection(poly.boundary)
    if hits.is_empty:
        return 0.0
    pts = [hits] if hits.geom_type == 'Point' else [g for g in getattr(hits, 'geoms', []) if g.geom_type == 'Point']
    s = [float(np.dot(np.array(p.coords[0]) - o, direction)) for p in pts]
    ahead = [v for v in s if v > 0] if inside else [v for v in s if v < 0]
    if not ahead:
        return 0.0
    return min(ahead) if inside else max(ahead)


def encode(d, median_font):
    """(centre[N,2], left[N], right[N]) for one Make Me a Hanzi-style stroke."""
    poly = outline_polygon(d)
    median = np.array([(x, 900 - y) for x, y in median_font], dtype=float)
    if len(median) == 1 or np.allclose(median[0], median[-1]):
        c = poly.centroid
        median = np.array([[c.x - 1, c.y], [c.x + 1, c.y]])
    # Extend the median along its end tangents to the outline's tips.
    start_dir = median[0] - median[1]
    end_dir = median[-1] - median[-2]
    start_dir /= np.linalg.norm(start_dir) or 1
    end_dir /= np.linalg.norm(end_dir) or 1
    ext = []
    for p, v in ((median[0], start_dir), (median[-1], end_dir)):
        inside = poly.contains(Point(p))
        ext.append(p + v * _hit(poly, p, v, inside))
    spine = np.vstack([ext[0], median, ext[1]])
    centre = resample(spine, cosine_spacing())
    tangents = _tangents(centre)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    left, right = [], []
    for p, n in zip(centre, normals):
        inside = poly.contains(Point(p))
        if inside:
            left.append(_hit(poly, p, n, True))
            right.append(_hit(poly, p, -n, True))
        else:
            left.append(0.0)
            right.append(0.0)
    return centre, np.array(left), np.array(right)


def decode_polygon(centre, left, right):
    """Outline polygon (screen coordinates) from the representation."""
    centre = np.asarray(centre, dtype=float)
    tangents = _tangents(centre)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    lw = np.clip(np.asarray(left), 0, None)[:, None]
    rw = np.clip(np.asarray(right), 0, None)[:, None]
    return np.vstack([centre + normals * lw, (centre - normals * rw)[::-1]])


def to_font(points):
    return [(float(x), float(900 - y)) for x, y in points]


def outline_path(centre, left, right):
    """Smoothed SVG outline in font coordinates, Make Me a Hanzi style.

    Quadratic curves through the midpoints of consecutive polygon vertices
    give a smooth outline without overshooting the sampled widths.
    """
    poly = to_font(decode_polygon(centre, left, right))
    n = len(poly)
    mid = [((poly[i][0] + poly[(i + 1) % n][0]) / 2, (poly[i][1] + poly[(i + 1) % n][1]) / 2) for i in range(n)]
    d = f'M {mid[-1][0]:.0f} {mid[-1][1]:.0f}'
    for i in range(n):
        d += f' Q {poly[i][0]:.0f} {poly[i][1]:.0f} {mid[i][0]:.0f} {mid[i][1]:.0f}'
    return d + ' Z'


def raster(polygons, size=256, frame=(-62.0, -124.0, 1148.0)):
    """Filled mask of screen-coordinate polygons (arrays or shapely)."""
    from PIL import Image, ImageDraw
    x0, y0, span = frame
    image = Image.new('1', (size, size), 0)
    draw = ImageDraw.Draw(image)
    for poly in polygons:
        geoms = getattr(poly, 'geoms', None)
        rings = ([np.asarray(g.exterior.coords) for g in geoms if g.geom_type == 'Polygon'] if geoms is not None
                 else [np.asarray(poly.exterior.coords)] if hasattr(poly, 'exterior') else [np.asarray(poly)])
        for ring in rings:
            if len(ring) >= 3:
                draw.polygon([((x - x0) / span * size, (y - y0) / span * size) for x, y in ring], fill=1)
    return np.asarray(image, dtype=bool)


def glyph_iou(polygons_a, polygons_b):
    """Raster IoU of two whole glyphs; robust to self-intersecting outlines."""
    a, b = raster(polygons_a), raster(polygons_b)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0
