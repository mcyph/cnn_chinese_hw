# -*- coding: utf-8 -*-
from collections import namedtuple, Counter
from heapq import heappush, heappushpop


from cnn_chinese_hw.stroke_tools.StrokeMetrics import (
    StrokeMetrics#, get_square_dist, get_angle_diff
)

from input.toolkit.benchmarking.benchmark import benchmark


Score = namedtuple('Score', [
    'angle_diff',
    'position_diff',
    #'vertice_diff',
    'size_diff',
    'stroke_diff',
    'total_points_penalty',
    'total_strokes_penalty'
])

Distance = namedtuple('Distance', [
    'to_index', 'dist', 'score'
])

Point = namedtuple('Point', [
    'x', 'y'
])

MAX_CANDS = 40
stroke_metrics = StrokeMetrics()


class Recognizer:
    def __init__(self, stroke_data):
        self.stroke_data = stroke_data

    #@benchmark()
    def get_L_cands(self, LStrokes, DOpts={}, process_strokes=True):
        self.DOpts = DOpts

        if process_strokes:
            LStrokes = stroke_metrics.get_L_strokes(LStrokes)

        num_points = len(LStrokes)
        if num_points < 6:
            plusminus = 0
        elif num_points < 8:
            plusminus = 1 # WARNING! ===================================================
        elif num_points < 12:
            plusminus = 2
        else:
            plusminus = 3

        LLens = list(range(num_points, num_points+plusminus+1))
        #print num_points, LLens
        LCands = []

        for len_ in LLens:
            #print 'LEN:', len_

            for key, ord_, i_LStrokes in self.stroke_data.iter_len(len_):
                self.IS_DEBUG_CHAR = ord_ == (
                    ord(self.DOpts.get('debug_char', 'a') or 'a')
                )

                if self.IS_DEBUG_CHAR:
                    print()
                    print(ord_, chr(ord_), len(LStrokes), len(i_LStrokes))

                dist = self.compare(LStrokes, i_LStrokes)
                if dist is None:
                    continue

                if len(LCands) > MAX_CANDS:
                    heappushpop(LCands, (-dist, ord_))
                else:
                    heappush(LCands, (-dist, ord_))

        return [(-i[0], i[1]) for i in sorted(LCands)]

    def compare(self, LStrokes1, LStrokes2):
        # {from index: (to_index, score), ...}
        DMap = {}
        DRevMap = {}

        # Only disqualify candidates which are far out on angle etc
        # if more than a certain number of strokes, as it can
        # currently cause more harm than good in that case.
        use_disqualify = max(
            len(LStrokes1), len(LStrokes2)
        ) > 5

        def process(i, j):
            score = self.get_score(
                LStrokes1[i], LStrokes2[j],
                abs(i-j),
                use_disqualify=use_disqualify
            )
            if score is None:
                return

            dist = self.calc_score(
                score
            )

            if not i in DMap or (dist < DMap[i].dist):

                if not j in DRevMap or (dist < DRevMap[j].dist):
                    if i in DMap:
                        del DRevMap[DMap[i].to_index]
                    if j in DRevMap:
                        del DMap[DRevMap[j].to_index]

                    #print 'SETTING:', i, j
                    DMap[i] = Distance(j, dist, score)
                    DRevMap[j] = Distance(i, dist, score)

        for i in range(len(LStrokes1)):
            for j in range(len(LStrokes2)):
                process(i, j)

        max_strokes = min(
            len(LStrokes1), len(LStrokes2)
        )

        for x in range(2):
            if len(DMap) == max_strokes:
                break

            for i in range(max_strokes):
                if i in DMap:
                    continue

                for j in range(max_strokes):
                    if j in DRevMap:
                        continue

                    process(i, j)

                    if len(DMap) == max_strokes:
                        break

        if self.IS_DEBUG_CHAR:
            for key, distance in list(DMap.items()):
                print(key, distance.score, distance.dist)

        return self.get_total_score(
            [v.score for v in list(DMap.values())],
            LStrokes1, LStrokes2
        )

    #====================================================#
    #                      Score Addition                      #
    #====================================================#

    def get_total_score(self, LScores,
                              LStrokes1, LStrokes2):

        if (
            (
                abs(len(LScores)-len(LStrokes2)) >
                len(LStrokes1) // 4
            )
            or not LScores
        ):
            return None

        score = sum([
            self.calc_score(score) for score in LScores
        ]) // len(LScores)

        score += abs(
            len(LStrokes1) - len(LStrokes2)
        ) * (
            self.DOpts.get(
                'total_strokes_penalty',
                self.DScoreTimes['total_strokes_penalty'][0]
            )
        ) #** (
        #    DScoreTimes['total_strokes_penalty'][1]
        #)

        return score

    #====================================================#
    #                 Vertice Difference Scoring               #
    #====================================================#

    DScoreTimes = {
        # 行 should be able to be written like the printed version
        'angle_diff': (1000, 1, 160),
        'position_diff': (1, 1, 350*350),
        #'vertice_diff': (0, 1, 250*250),
        'size_diff': (3, 1, 500*500),
        'stroke_diff': (30000, 1, 65535),
        'total_points_penalty': (30000, 1, 10),
        'total_strokes_penalty': (30000, 1, 1000)
    }

    def get_score(self, s1, s2, stroke_diff,
                        DScoreTimes=None,
                        use_disqualify=True):

        DScoreTimes = DScoreTimes or self.DScoreTimes
        DOpts = self.DOpts

        D = {
            'total_strokes_penalty': None
        }

        def scale(field, val):
            times, power, cutoff = DScoreTimes[field]
            times = DOpts.get(field, times)

            if use_disqualify:
                if field == 'size_diff':
                    cutoff = 1000 + (
                        min(s1.width, s2.width) *
                        min(s1.height, s2.height)
                    ) * 1.4
                    #print cutoff, val

                if val > cutoff:
                    return False

            if times != 1:
                val *= times
            #if power != 1:
            #    val **= power

            D[field] = val
            return True

        if not scale('angle_diff', self.get_angle_diff(s1, s2)): return None
        if not scale('position_diff', self.get_position_diff(s1, s2)): return None
        #if not scale('vertice_diff', self.get_vertice_diff(s1, s2)): return None
        if not scale('size_diff', self.get_size_diff(s1, s2)): return None
        if not scale('stroke_diff', stroke_diff): return None
        if not scale('total_points_penalty', self.get_total_points_penalty(s1, s2)): return None

        return Score(**D)

    def calc_score(self, score):
        r = 0

        for field in score._fields:
            if field == 'total_strokes_penalty':
                continue

            val = getattr(score, field)
            r += val

        return r

    def get_angle_diff(self, s1, s2):
        r = 0

        for angle1, angle2 in zip(s1.LAngles, s2.LAngles):
            r += get_angle_diff(angle1, angle2)

        #print r
        return int(r//4)

        #return abs(s1.angle-s2.angle)

        #LRtn = []

        #for x, angle in enumerate(s1.LAngles):
        #    LRtn.append(get_angle_diff(
        #        angle,
        #        self.get_scaled_angle(
        #            x / float(len(s1.LAngles)),
        #            s2
        #        )
        #    ))

        return sum(LRtn)

    def get_position_diff(self, s1, s2):
        return get_square_dist(
            s1.x, s1.y, s2.x, s2.y
        )

    def get_vertice_diff(self, s1, s2):
        return 0 # HACK!

        r = 0
        LPoints1 = s1.LPoints

        for i in range(len(LPoints1)-1):
            x1_1, y1_1 = LPoints1[i]
            x2_1, y2_1 = LPoints1[i+1]

            point_1 = self.get_scaled_point(
                i / float(len(LPoints1)-1),
                s2, s1
            )
            point_2 = self.get_scaled_point(
                (i+1) / float(len(LPoints1)-1),
                s2, s1
            )

            #print x1_1, y1_1, x2_1, y2_1
            #print point_1, point_2

            d1 = get_square_dist(x1_1, y1_1, point_1.x, point_1.y)
            d2 = get_square_dist(x2_1, y2_1, point_2.x, point_2.y)

            #d3 = get_square_dist(x1_1, y1_1, x2_1, y2_1)
            #d4 = get_square_dist(point_1, point_1, point_2, point_2)

            r += d1 + d2 #+ abs(d3 - d4)
            #print dist, (x1_1, y1_1, x2_1, y2_1), (x1_2, y1_2, x2_2, y2_2)

        return r

    def get_size_diff(self, s1, s2):
        return (
            abs(s1.width-s2.width) *
            abs(s1.height-s2.height)
        )

    def get_total_strokes_penalty(self, LStrokes1, LStrokes2):
        return abs(
            len(LStrokes1) -
            len(LStrokes2)
        )

    def get_total_points_penalty(self, s1, s2):
        return abs(
            len(s1.LPoints) -
            len(s2.LPoints)
        )

    #====================================================#
    #                   Scaled Points/Angles                   #
    #====================================================#

    def get_scaled_point(self, percent, stroke, other_stroke):
        """
        Get a scaled point
        e.g. if there's only two points and 50% (0.5)
        is asked for, it'll go in between of the two

        FIXME: Scale for size, as well! =======================================================================
        """
        LPoints = stroke.LPoints
        len_ = len(LPoints)-1

        get_from = len_ * percent
        higher_amount = get_from-int(get_from)
        lower_amount = 1.0 - higher_amount
        get_from = int(get_from) # rounds down
        #print '%:', percent, 'GF:', get_from, lower_amount, higher_amount, get_from == len_

        point1 = LPoints[get_from]
        if get_from == len_:
            return Point(point1[0], point1[1])

        point2 = LPoints[get_from + 1]

        x_times = other_stroke.x/float(stroke.x or 1)
        y_times = other_stroke.y/float(stroke.y or 1)
        times = x_times if x_times > y_times else y_times

        #print times, other_stroke.x, other_stroke.y, stroke.x, stroke.y

        return Point(
            (
                (point1[0] * lower_amount) +
                (point2[0] * higher_amount)
            ) * times,

            (
                (point1[1] * lower_amount) +
                (point2[1] * higher_amount)
            ) * times
        )

    def get_scaled_angle(self, percent, stroke):
        # CHECK THIS WORKS WITH ALL ANGLES! ======================================================================
        LAngles = stroke.LAngles
        len_ = len(LAngles)-1

        get_from = len_ * percent
        higher_amount = get_from-int(get_from)
        lower_amount = 1.0 - higher_amount
        get_from = int(get_from) # rounds down

        try:
            angle1 = LAngles[get_from]
        except:
            print(LAngles, get_from)
            raise

        if get_from == len_:
            return angle1

        angle2 = LAngles[get_from + 1]

        return (
            (angle1 * lower_amount) +
            (angle2 * higher_amount)
        )


def run():
    from handwriting.parse_data import StrokeData


    print('Getting stroke data...')
    sd = StrokeData()

    print('Creating Recognizer...')
    r = Recognizer(sd)


    while 1:
        inp = input("Enter a character: ")
        lookup_ord = ord(inp.decode('utf-8')[0]) if inp else ord('车')

        print('Recognizing:', lookup_ord, sd.get(lookup_ord)[0])
        print(r.get_L_cands)

        for (score, ord_) in r.get_L_cands(
            sd.get(lookup_ord)[0], process_strokes=False

        ):
            print(score, ord_, chr(ord_))


if __name__ == '__main__':
    run()
