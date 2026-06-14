# -*- coding: utf-8 -*-
"""Parser for the Make Me a Hanzi ``graphics.txt`` stroke corpus.

``graphics.txt`` is a newline-separated list of JSON objects, one per
character, each with a ``"medians"`` field: a list of strokes, where every
stroke is a poly-line of ``[x, y]`` control points tracing the centre line of
the stroke. That centre line is effectively a synthetic *pen trajectory*, so it
slots straight into this project's online (stroke-based) pipeline alongside the
Tomoe and KanjiVG parsers.

Coordinate system
-----------------
Make Me a Hanzi glyphs live in a 1024-unit em square using *font* coordinates,
where the y-axis points **up**. Tomoe / KanjiVG (and our rasteriser) use
*screen* coordinates, where y points **down**. We therefore apply Make Me a
Hanzi's own documented render transform ``matrix(1 0 0 -1 0 900)`` i.e.
``y_screen = 900 - y_font``; without it every character would be rendered
vertically mirrored. ``points_normalized`` downstream re-bases the bounding box,
so only the flip (not the absolute offset/scale) matters.

License
-------
``graphics.txt`` is derived from Arphic fonts and distributed under the **Arphic
Public License** (commercial use permitted; not GPL). See
``cnn_chinese_hw/data/makemeahanzi/`` for the license text and attribution, and
the repo-level ``THIRD_PARTY_DATA.md``.
"""

import json
import os

from cnn_chinese_hw.get_package_dir import get_package_dir

# Make Me a Hanzi's render transform is `matrix(1 0 0 -1 0 900)`, i.e. flip the
# y-axis about y=900 to go from font (y-up) to screen (y-down) coordinates.
_Y_FLIP_ABOUT = 900

DEFAULT_PATH = f'{get_package_dir()}/data/makemeahanzi/graphics.txt'

_DOWNLOAD_URL = (
    'https://raw.githubusercontent.com/skishore/makemeahanzi/master/graphics.txt'
)


def iter_makemeahanzi_data(path=DEFAULT_PATH):
    """Yield ``(character_ordinal, [stroke_as_list_of_(x, y), ...])`` for every
    character in the Make Me a Hanzi ``graphics.txt`` corpus.

    Raises ``FileNotFoundError`` (with a hint to :func:`download`) if the data
    file is missing, so callers can choose to skip the source gracefully.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Make Me a Hanzi data not found at {path}. "
            f"Run `python -m cnn_chinese_hw.parse_data.iter_makemeahanzi_data "
            f"--download` (or `download()`) to fetch graphics.txt."
        )

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            char = obj.get('character')
            medians = obj.get('medians')
            # Skip entries without a single-codepoint character or without
            # centre-line data (a handful of components have empty medians).
            if not char or len(char) != 1 or not medians:
                continue

            strokes_list = [
                [(x, _Y_FLIP_ABOUT - y) for x, y in stroke]
                for stroke in medians
                if stroke
            ]
            if not strokes_list:
                continue

            yield ord(char), strokes_list


def download(path=DEFAULT_PATH, url=_DOWNLOAD_URL):
    """Fetch ``graphics.txt`` (~30 MB) to ``path`` if it is not already there."""
    import urllib.request

    if os.path.exists(path):
        print(f"Already present: {path}")
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)
    print("Done.")
    return path


if __name__ == '__main__':
    import sys

    if '--download' in sys.argv:
        download()
    else:
        count = 0
        for ord_, strokes in iter_makemeahanzi_data():
            count += 1
            if count <= 3:
                print(ord_, chr(ord_), [len(s) for s in strokes])
        print(f"total characters: {count}")
