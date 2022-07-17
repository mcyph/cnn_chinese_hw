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
    def combine_similar_vertices(self, LStrokes):
        """
        "Chains" strokes if they're very close to each other.

        e.g. [start]====[end][start]====[end]

        doesn't allow [start]====[end][end]====[start]
        configurations as I think that'd cause more problems
        than it's worth...
        """
        return LStrokes # HACK! =============================================================

        DMatch = {}
        DMatchRev = {}

        for x, LStroke1 in enumerate(LStrokes):
            for y, LStroke2 in enumerate(LStrokes):
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
        #    print x, y, LStrokes[x], LStrokes[y]
        #print DChain, DChainRev, LStrokes
        #print DMatch

        DRtn = {}
        SXNotUsed = set(range(len(LStrokes)))

        for x in list(DChain.keys()):
            if x in DChainRev:
                continue

            LOut = DRtn[x] = LStrokes[x][:]

            while 1:
                SXNotUsed.remove(x)

                if not x in DChain:
                    break

                y = DChain.pop(x)
                LOut.extend(LStrokes[y])

                x = y

        for x in sorted(SXNotUsed):
            DRtn[x] = LStrokes[x]

        return [
            v for k, v in sorted(DRtn.items())
        ]

    def get_L_strokes(self, LStrokes,
                            normalize_points=True):

        if normalize_points:
            LStrokes = points_normalized(LStrokes)

        return_list = []

        for LPoints in LStrokes:
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

    def get_L_brensenham(self, LStroke):
        """
        Convert points to lines using
        the Brensenham algorithm
        """
        return_list = []

        for x in range(len(LStroke)-1):
            return_list.extend(brensenham_line(*(
                LStroke[x]+LStroke[x+1]
            )))

        new_return_list = []
        x = 0
        while 1:
            if x >= len(return_list):
                break

            new_return_list.append(return_list[x])
            x += (len(return_list)//4) or 4 #VERTICE_EVERY

        if len(new_return_list) < 2:
            new_return_list.append(LStroke[-1])
        return new_return_list


if __name__ == '__main__':
    sm = StrokeMetrics()

    print(sm.combine_similar_vertices([
        [(0, 0), (0, 1)],
        [(0, 1), (0, 2)],
        [(0, 2), (0, 3)]
    ]))

    LTry = [
        [(0, 74), (80, 875)],
        [(214, 40), (982, 214)],
        [(170, 1000), (929, 948)]
    ]


    print(sm.combine_similar_vertices(LTry))
    print(sm.get_L_strokes(LTry))
