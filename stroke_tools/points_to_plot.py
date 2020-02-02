from cnn_chinese_hw.stroke_tools.brensenham_line import brensenham_line
from cnn_chinese_hw.stroke_tools.xiaolinwu import line


def points_to_plot(LPoints):
    LRtn = []

    for LStrokes in LPoints:
        i = 0
        LItem = []
        last_stroke = None
        for stroke in LStrokes:
            if i > 0:
                LExtend = [(int(x), int(y)) for x, y in brensenham_line(
                    last_stroke[0],
                    last_stroke[1],
                    stroke[0],
                    stroke[1]
                )]
                LItem.extend(LExtend)

            last_stroke = stroke
            i += 1
        LRtn.append(LItem)

    return LRtn


def draw_in_place(array, LPoints):
    for LStrokes in LPoints:
        i = 0
        last_stroke = None
        for stroke in LStrokes:
            if i > 0:
                #print(stroke, last_stroke)
                line(
                    array,
                    last_stroke[0], last_stroke[1],
                    stroke[0], stroke[1]
                )

            last_stroke = stroke
            i += 1
