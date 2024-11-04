from cnn_chinese_hw.stroke_tools.brensenham_line import brensenham_line
from cnn_chinese_hw.stroke_tools.xiaolinwu import line, fading_line


def points_to_plot(strokes_list):
    return_list = []

    for stroke_list in strokes_list:
        i = 0
        item_list = []
        last_stroke = None
        for stroke in stroke_list:
            if i > 0:
                LExtend = [(int(x), int(y)) for x, y in brensenham_line(
                    last_stroke[0],
                    last_stroke[1],
                    stroke[0],
                    stroke[1]
                )]
                item_list.extend(LExtend)

            last_stroke = stroke
            i += 1
        return_list.append(item_list)

    return return_list


def draw_faded_brensenham_lines(a, strokes_list):
    for j, stroke_list in enumerate(strokes_list):
        i = 0
        item_list = []
        last_stroke = None
        for stroke in stroke_list:
            if i > 0:
                LExtend = [(int(x), int(y)) for x, y in brensenham_line(
                    last_stroke[0],
                    last_stroke[1],
                    stroke[0],
                    stroke[1]
                )]
                item_list.extend(LExtend)

            last_stroke = stroke
            i += 1

        for i, (x, y) in enumerate(item_list):
            if len(item_list) == 1:
                stroke_pc = 1.0
            else:
                stroke_pc = i / (len(item_list) - 1)

            # Output the start and end of the strokes respectively
            # as separate layers, so as to prioritize correct
            # stroke directions. It may pay to tune the 0.2/0.8
            # to either be more or less lenient, depending on purposes.
            opacity1 = 0.2 + (1.0 - stroke_pc) * 0.8
            opacity2 = 0.2 + stroke_pc * 0.8

            # For the third layer, output darker colours for the
            # first strokes so as to prioritize the correct stroke
            # order (to a degree)
            # I haven't added the inverse direction, as it might
            # be useful to allow predictions for a few less strokes.
            strokes_pc = (
                j / (len(strokes_list)-1)
                if len(strokes_list) > 1
                else 1.0
            )

            a[y, x, 0] = max(a[y, x, 0], opacity1*255)
            a[y, x, 1] = max(a[y, x, 1], opacity2*255)
            a[y, x, 2] = max(a[y, x, 2], (1.0-strokes_pc)*255)


def draw_in_place(array, strokes_list, reverse_direction=False):
    for stroke_list in strokes_list:
        i = 0
        last_stroke = None
        for stroke in stroke_list:
            if not i:
                i += 1
                last_stroke = stroke
                continue

            if reverse_direction:
                x1, y1, x2, y2 = (
                    stroke[0], stroke[1],
                    last_stroke[0], last_stroke[1]
                )
                if len(stroke_list) > 2:
                    pc = 1.0-((i - 1) / (len(stroke_list) - 2))
            else:
                x1, y1, x2, y2 = (
                    last_stroke[0], last_stroke[1],
                    stroke[0], stroke[1]
                )
                if len(stroke_list) > 2:
                    pc = (i - 1) / (len(stroke_list) - 2)

            if i > 0 and len(stroke_list) == 2:
                # Single set of points: fade as a whole
                #print(stroke, last_stroke)
                fading_line(array, x1, y1, x2, y2)
            elif i > 0:
                # Multiple points: fade for each part
                # This might play up if there's a really
                # long portion with a short portion, but
                # hopefully it'll work ok most of the time

                line(
                    array, x1, y1, x2, y2,
                    opacity=0.2+(1.0-pc)*0.8
                )

            last_stroke = stroke
            i += 1
