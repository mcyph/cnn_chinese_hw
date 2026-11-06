"""AnimCJK regional stroke outlines (Japanese and Taiwan Traditional).

AnimCJK keeps a separate glyph per region, with that region's letterforms,
stroke order and stroke count (e.g. Taiwan 艹 has 4 strokes and 馬 starts with
the vertical). Its ``graphics*.txt`` files use Make Me a Hanzi's format and
coordinate frame (one JSON object per line with ``strokes`` outlines and
``medians``, 1024-unit em, y up, render transform ``matrix(1 0 0 -1 0 900)``),
so ``display_strokes`` reads them with the same loader.

Only the Japanese kanji and Taiwan files are used. The kana file is not: its
looped kana are split into extra medians (あ has 4, な 5), which would teach
the wrong stroke count.

The data is derived from Make Me a Hanzi and the Arphic PL fonts, under the
Arphic Public License; see ``cnn_chinese_hw/data/animcjk/`` and
THIRD_PARTY_DATA.md. The files are not checked into Git. Fetch the pinned,
checksum-verified release with
``python -m cnn_chinese_hw.parse_data.animcjk_data --download``.
"""
import hashlib
import os

from cnn_chinese_hw.get_package_dir import get_package_dir

COMMIT = '1c8c8ef45aa9dfe1e1129cb5462ddf59a50a65ce'   # 2026-11-02
_DIR = f'{get_package_dir()}/data/animcjk'
_URL = f'https://raw.githubusercontent.com/parsimonhi/animCJK/{COMMIT}/{{name}}'
FILES = {
    'ja': ('graphicsJa.txt', '3b2d95465152fb2a7e108774219e969c43af14f1e51e66498bc96522a674cfeb'),
    'zh-TW': ('graphicsZhHant.txt', 'e2e999f614381392d8565aa1ea4e0c61cdfa945c2811cd70309e00742d3443a8'),
}
JA_PATH = f'{_DIR}/{FILES["ja"][0]}'
HANT_PATH = f'{_DIR}/{FILES["zh-TW"][0]}'


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def download(directory=_DIR):
    """Fetch the pinned files, refusing any whose checksum does not match."""
    import urllib.request

    os.makedirs(directory, exist_ok=True)
    for name, expected in FILES.values():
        path = os.path.join(directory, name)
        if os.path.exists(path) and _sha256(path) == expected:
            print(f'Already present: {path}')
            continue
        partial = f'{path}.partial'
        print(f'Downloading {name} @ {COMMIT[:7]} -> {path}')
        urllib.request.urlretrieve(_URL.format(name=name), partial)
        actual = _sha256(partial)
        if actual != expected:
            os.remove(partial)
            raise ValueError(f'{name}: sha256 {actual} != pinned {expected}')
        os.replace(partial, path)
    print('Done.')


if __name__ == '__main__':
    import sys

    if '--download' in sys.argv:
        download()
    else:
        for variant, (name, _) in FILES.items():
            path = os.path.join(_DIR, name)
            print(variant, path, 'present' if os.path.exists(path) else 'missing')
