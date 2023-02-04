from svg.path import Move, CubicBezier, parse_path
from cnn_chinese_hw.get_package_dir import get_package_dir


def iter_kanjivg_data(path=f'{get_package_dir()}/'
                           f'data/validation/kanjivg.xml'):
    """
    Yields (character ordinal, [stroke as a list of x, y points, ...])
    for every character. Does not get all data from KanjiVG,
    like the radicals etc as I'm mainly using this for verification.
    """
    hex_ord = None
    LStrokes = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line.startswith('<kanji id="kvg:kanji_'):
                if hex_ord:
                    yield hex_ord, LStrokes

                hex_ord = int(line.partition(
                    '<kanji id="kvg:kanji_'
                )[-1].strip('">'), 16)
                LStrokes = []
            elif line.startswith('<path id="kvg:'):
                path = line.partition(' d="')[-1].strip('"/>')
                path = parse_path(path)
                LPath = _get_L_path(path)
                LStrokes.append(LPath)

    yield hex_ord, LStrokes


def _get_L_path(path):
    """
    Strictly speaking, these are cubic bezier curves,
    but I think it might be good enough to just use
    the control points
    """
    out_list = []
    for i_path in path:
        if isinstance(i_path, Move):
            pass
        elif isinstance(i_path, CubicBezier):
            out_list.extend((
                (i_path.start.real, i_path.start.imag),
                (i_path.control1.real, i_path.control1.imag),
                (i_path.control2.real, i_path.control2.imag),
                (i_path.end.real, i_path.end.imag)
            ))
        else:
            raise Exception("Unsupported:", i_path)
    return out_list


if __name__ == '__main__':
    for ord_, LPaths in iter_kanjivg_data():
        print(ord_, LPaths)
