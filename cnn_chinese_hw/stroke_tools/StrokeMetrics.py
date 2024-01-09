from collections import namedtuple

from .brensenham_line import brensenham_line
from .points_normalized import points_normalized


Stroke = namedtuple('Stroke', [
    'x', 'y',
    'width', 'height',
    'LPoints',
    'LLengths'
])

VERTICE_EVERY = 400
MAX_DIFF = 0 #20*20 #300*300


class StrokeMetrics:
    def combine_similar_vertices(self, strokes_list):
        """
        "Chains" strokes if they're very close to each other.

        e.g. [start]====[end][start]====[end]

        doesn't allow [start]====[end][end]====[start]
        configurations as I think that'd cause more problems
        than it's worth...
        """
        return strokes_list # HACK! =============================================================

        DMatch = {}
        DMatchRev = {}

        for x, LStroke1 in enumerate(strokes_list):
            for y, LStroke2 in enumerate(strokes_list):
                if x == y:
                    continue

                last_point_1 = LStroke1[-1]
                first_point_2 = LStroke2[0]

                dist = get_square_dist(*(
                    last_point_1 + first_point_2
                ))

                if dist > MAX_DIFF:
                    # ignore if too far away
                    continue
                elif x in DMatch and DMatch[x][0] <= dist:
                    # don't update if any previous matches
                    # were closer than this one
                    continue
                elif y in DMatchRev and DMatchRev[y][0] <= dist:
                    continue

                DMatch[x] = (dist, y)
                DMatchRev[y] = (dist, x)

        DChain = {}
        DChainRev = {}

        for x, (dist, y) in sorted(
            list(DMatch.items()),
            key=lambda i: i[1][0]
        ):
            if y in DChainRev: # SMALLER SCORE WARNING! ================================
                continue
            elif y in DMatch and DMatch[y][1] == x and DMatch[y][0] <= dist:
                continue

            DChain[x] = y
            DChainRev[y] = x

        #for x, y in DChain.items():
        #    print x, y, strokes_list[x], strokes_list[y]
        #print DChain, DChainRev, strokes_list
        #print DMatch

        return_dict = {}
        SXNotUsed = set(range(len(strokes_list)))

        for x in list(DChain.keys()):
            if x in DChainRev:
                continue

            out_list = return_dict[x] = strokes_list[x][:]

            while 1:
                SXNotUsed.remove(x)

                if not x in DChain:
                    break

                y = DChain.pop(x)
                out_list.extend(strokes_list[y])

                x = y

        for x in sorted(SXNotUsed):
            return_dict[x] = strokes_list[x]

        return [
            v for k, v in sorted(return_dict.items())
        ]

    def get_L_strokes(self, strokes_list,
                            normalize_points=True):

        if normalize_points:
            strokes_list = points_normalized(strokes_list)

        return_list = []

        for LPoints in strokes_list:
            min_x = min(i[0] for i in LPoints)
            min_y = min(i[1] for i in LPoints)
            max_x = max(i[0] for i in LPoints)
            max_y = max(i[1] for i in LPoints)

            return_list.append(Stroke(
                x=min_x + ((max_x-min_x) / 2),
                y=min_y + ((max_y-min_y) / 2),
                width=max_x-min_x,
                height=max_y-min_y,
                LPoints=LPoints,
                LLengths=None # FIXME! ===================================================
            ))

        return return_list

    def get_L_brensenham(self, stroke_list):
        """
        Convert points to lines using
        the Brensenham algorithm
        """
        return_list = []

        for x in range(len(stroke_list)-1):
            return_list.extend(brensenham_line(*(
                stroke_list[x]+stroke_list[x+1]
            )))

        new_return_list = []
        x = 0
        while 1:
            if x >= len(return_list):
                break

            new_return_list.append(return_list[x])
            x += (len(return_list)//4) or 4 #VERTICE_EVERY

        if len(new_return_list) < 2:
            new_return_list.append(stroke_list[-1])
        return new_return_list


if __name__ == '__main__':
    sm = StrokeMetrics()

    print(sm.combine_similar_vertices([
        [(0, 0), (0, 1)],
        [(0, 1), (0, 2)],
        [(0, 2), (0, 3)]
    ]))

    try_list = [
        [(0, 74), (80, 875)],
        [(214, 40), (982, 214)],
        [(170, 1000), (929, 948)]
    ]


    print(sm.combine_similar_vertices(try_list))
    print(sm.get_L_strokes(try_list))
