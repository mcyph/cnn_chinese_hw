#!/usr/bin/env python3
"""Emit specimen-strokes JSON blocks for Han characters, sourced from
Make Me a Hanzi `medians` (centerline trajectories), transformed to match
the frontend StrokeOrder component (centerline stroked at width 8 in a
~109-unit y-down viewBox). MMH lives in a 1024 em square, y-up, with the
documented render flip matrix(1 0 0 -1 0 900): screen_y = 900 - font_y.
"""
import json, sys, pathlib
GRAPHICS = "/home/david/dev/langlynx/cnn_chinese_hw/cnn_chinese_hw/data/makemeahanzi/graphics.txt"
SCALE = 109/1024.0
def tf(x, y):
    return round(x*SCALE, 1), round((900 - y)*SCALE, 1)
def median_to_path(median):
    pts = [tf(x, y) for x, y in median]
    if not pts: return ""
    d = f"M {pts[0][0]} {pts[0][1]}"
    for X, Y in pts[1:]:
        d += f" L {X} {Y}"
    return d
def main(chars):
    want = set(chars); found = {}
    with open(GRAPHICS, encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            # cheap prefilter
            if not any(c in line[:30] for c in want): continue
            try: obj = json.loads(line)
            except Exception: continue
            ch = obj.get("character")
            if ch in want and ch not in found:
                found[ch] = obj
                if len(found) == len(want): break
    out = []
    for ch in chars:
        obj = found.get(ch)
        if not obj or not obj.get("medians"):
            out.append(f"# no stroke data for {ch}"); continue
        strokes = [median_to_path(m) for m in obj["medians"] if m]
        block = {"glyph": ch, "strokes": strokes,
                 "viewBox": {"width": 109, "height": 109}}
        out.append("```specimen-strokes\n" + json.dumps(block, ensure_ascii=False) + "\n```")
    print("\n\n".join(out))
if __name__ == "__main__":
    main(sys.argv[1:])
