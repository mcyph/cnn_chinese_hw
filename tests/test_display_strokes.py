import json

import pytest

from cnn_chinese_hw.parse_data.display_strokes import display_strokes
from cnn_chinese_hw.parse_data.kanjivg_strokes import kanjivg_strokes


class FakeStrokes:
    def __init__(self):
        # StrokeMetrics tuples: the fifth item is the original trajectory.
        self.data = {
            'ja': [[(0, 0, 0, 0, [(10, 20), (30, 40)], [])]],
            'zh': [[(0, 0, 0, 0, [(11, 21), (31, 41)], [])]],
            'zh-TW': [[(0, 0, 0, 0, [(12, 22), (32, 42)], [])]],
        }

    def get(self, ord_, keys_list):
        assert ord_ == ord('一')
        return self.data.get(keys_list[0], [])


def outlines(path, count=1, medians=None):
    """A Make Me a Hanzi / AnimCJK style corpus holding 一 (y up)."""
    record = {'character': '一', 'strokes': ['M 0 0 L 10 10 Z'] * count,
              'medians': medians or [[[0, 900], [10, 890]]] * count}
    path.write_text(json.dumps(record, ensure_ascii=False) + '\n', encoding='utf-8')
    return path


KANJIVG = """<?xml version="1.0" encoding="UTF-8"?>
<kanjivg>
<kanji id="kvg:kanji_04e00">
<g id="kvg:04e00" kvg:element="一">
\t<path id="kvg:04e00-s1" kvg:type="㇐" d="M11,54.25c3.19,0.62,6.25,0.75,9.73,0.5c20.64-1.5,50.39-5.12,68.58-5.24c3.6-0.02,5.77,0.24,7.57,0.49"/>
</g>
</kanji>
<kanji id="kvg:kanji_04e01">
<g id="kvg:04e01">
\t<path id="kvg:04e01-s1" kvg:type="㇐" d="M10,20L90,20"/>
\t<path id="kvg:04e01-s2" kvg:type="㇚/㇑" d="M50,20L50,90L40,80"/>
</g>
</kanji>
</kanjivg>
"""


@pytest.fixture
def paths(tmp_path):
    """Every corpus path absent until a test writes it, so the real data on
    the host never leaks into a test."""
    class Paths:
        mmh = tmp_path / 'graphics.txt'
        kanjivg = tmp_path / 'kanjivg.xml'
        ja = tmp_path / 'graphicsJa.txt'
        hant = tmp_path / 'graphicsZhHant.txt'
        model = tmp_path / 'stroke_outline.pt'

        def show(self, variant, data=None, char='一'):
            return display_strokes(char, variant, data or FakeStrokes(), self.mmh,
                                   self.kanjivg, self.ja, self.hant, self.model)

        def write_kanjivg(self):
            self.kanjivg.write_text(KANJIVG, encoding='utf-8')
    return Paths()


def test_outline_source_keeps_outline_and_median_together(paths):
    outlines(paths.mmh)
    result = paths.show('zh')
    assert result['source'] == 'makemeahanzi'
    assert result['transform'] == 'translate(0 900) scale(1 -1)'
    assert result['strokes'] == [{'outline': 'M 0 0 L 10 10 Z', 'points': [[0, 900], [10, 890]]}]
    assert paths.show('ja')['variant'] == 'ja'
    assert paths.show('zh-TW')['variant'] == 'zh-TW'


def test_simplified_trusts_make_me_a_hanzi_without_the_tomoe_screen(paths):
    # A different count or direction from Tomoe zh no longer demotes it.
    outlines(paths.mmh, count=2)
    assert paths.show('zh')['source'] == 'makemeahanzi'
    outlines(paths.mmh, medians=[[[10, 890], [0, 900]]])
    assert paths.show('zh')['source'] == 'makemeahanzi'
    # Characters Tomoe zh lacks are covered too.
    data = FakeStrokes()
    data.data.pop('zh')
    assert paths.show('zh', data)['source'] == 'makemeahanzi'


def test_tomoe_is_the_last_resort_and_never_another_locales(paths):
    data = FakeStrokes()
    assert paths.show('zh', data)['source'] == 'tomoe'
    assert paths.show('ja', data)['strokes'][0]['points'][0] == (10, 20)
    assert paths.show('zh-TW', data)['strokes'][0]['points'][0] == (12, 22)
    assert paths.show('ko', data) is None
    data.data.pop('zh-TW')
    assert paths.show('zh-TW', data) is None


def test_outline_file_provisioned_after_start_is_seen(paths):
    assert paths.show('zh')['source'] == 'tomoe'
    outlines(paths.mmh)
    assert paths.show('zh')['source'] == 'makemeahanzi'


def test_traditional_screens_make_me_a_hanzi_against_tomoe(paths):
    outlines(paths.mmh, count=2)
    assert paths.show('zh-TW')['source'] == 'tomoe'
    outlines(paths.mmh, medians=[[[10, 890], [0, 900]]])
    assert paths.show('zh-TW')['source'] == 'tomoe'
    outlines(paths.mmh)
    assert paths.show('zh-TW')['source'] == 'makemeahanzi'


def test_traditional_prefers_taiwan_animcjk_even_without_tomoe(paths):
    outlines(paths.mmh)
    outlines(paths.hant, medians=[[[5, 900], [15, 890]]])
    result = paths.show('zh-TW')
    assert result['source'] == 'animcjk' and result['variant'] == 'zh-TW'
    assert result['transform'] == 'translate(0 900) scale(1 -1)'
    assert result['strokes'][0]['points'][0] == [5, 900]
    data = FakeStrokes()
    data.data.pop('zh-TW')
    assert paths.show('zh-TW', data)['source'] == 'animcjk'
    # Other locales never use the Taiwan glyphs.
    assert paths.show('zh')['source'] == 'makemeahanzi'


def test_japanese_prefers_kanjivg_over_outlines_and_tomoe(paths):
    outlines(paths.mmh)
    paths.write_kanjivg()
    result = paths.show('ja')
    assert result['source'] == 'kanjivg' and result['variant'] == 'ja'
    assert result['viewBox'] == [0, 0, 1000, 1000]
    [stroke] = result['strokes']
    assert stroke['kind'] == '㇐'
    # Sampled in the 1000-unit frame, left to right.
    assert stroke['points'][0] == [round(11 * 1000 / 109, 1), round(54.25 * 1000 / 109, 1)]
    assert stroke['points'][-1][0] > 850 and len(stroke['points']) > 10
    # Chinese-font outlines are used again only when KanjiVG lacks the
    # character; other locales never see KanjiVG.
    paths.kanjivg.unlink()
    assert paths.show('ja')['source'] == 'makemeahanzi'
    paths.write_kanjivg()
    assert paths.show('zh')['source'] == 'makemeahanzi'


def test_japanese_animcjk_is_used_only_where_it_agrees_with_kanjivg(paths):
    paths.write_kanjivg()
    # Left to right, one stroke: agrees with KanjiVG's 一.
    outlines(paths.ja, medians=[[[100, 400], [900, 420]]])
    result = paths.show('ja')
    assert result['source'] == 'animcjk' and result['variant'] == 'ja'
    # Drawn right to left: KanjiVG wins.
    outlines(paths.ja, medians=[[[900, 420], [100, 400]]])
    assert paths.show('ja')['source'] == 'kanjivg'
    # A different stroke count: KanjiVG wins.
    outlines(paths.ja, count=2, medians=[[[100, 400], [500, 410]], [[500, 410], [900, 420]]])
    assert paths.show('ja')['source'] == 'kanjivg'
    # Without KanjiVG there is nothing to check it against, and it is used.
    paths.kanjivg.unlink()
    assert paths.show('ja')['source'] == 'animcjk'


def test_kanjivg_covers_japanese_characters_tomoe_lacks_and_keeps_corners(paths):
    data = FakeStrokes()
    data.data.pop('ja')
    paths.write_kanjivg()
    assert paths.show('ja', data)['source'] == 'kanjivg'
    paths.kanjivg.unlink()
    assert paths.show('ja', data) is None
    paths.write_kanjivg()
    hook = kanjivg_strokes('丁', paths.kanjivg)['strokes'][1]
    assert hook['kind'] == '㇚'
    scale = 1000 / 109
    # The corner at the bottom of the vertical is an exact sample.
    assert [round(50 * scale, 1), round(90 * scale, 1)] in hook['points']
    assert kanjivg_strokes('二', paths.kanjivg) is None


def test_stroke_model_replaces_tomoe_keeping_its_order_and_count(paths):
    torch = pytest.importorskip('torch')
    from cnn_chinese_hw.stroke_model import representation as R
    from cnn_chinese_hw.stroke_model.model import StrokeOutlineModel

    torch.manual_seed(0)
    torch.save({'state_dict': StrokeOutlineModel(d_model=256).state_dict(), 'n': R.N}, paths.model)
    data = FakeStrokes()
    data.data['zh-TW'] = [[(0, 0, 0, 0, [(100, 500), (900, 500)], []), (0, 0, 0, 0, [(500, 100), (500, 900)], [])]]
    result = paths.show('zh-TW', data)
    assert result['source'] == 'generated' and result['variant'] == 'zh-TW'
    assert result['transform'] == 'translate(0 900) scale(1 -1)'
    assert len(result['strokes']) == 2
    assert all(s['outline'].startswith('M ') and s['outline'].endswith('Z') and len(s['points']) == R.N
               for s in result['strokes'])
    # Better sources still win.
    outlines(paths.hant)
    assert paths.show('zh-TW', data)['source'] == 'animcjk'
