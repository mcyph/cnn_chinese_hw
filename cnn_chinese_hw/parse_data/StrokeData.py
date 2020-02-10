# -*- coding: utf-8 -*-
from cnn_chinese_hw.get_package_dir import get_package_dir
from cnn_chinese_hw.parse_data import iter_tomoe_data
from cnn_chinese_hw.stroke_tools.StrokeMetrics import StrokeMetrics

stroke_metrics = StrokeMetrics()


class StrokeData:
    def __init__(self):
        self.D = {
            'ja': self.get_stroke_data(
                f'{get_package_dir()}/data/handwriting-ja.xml'
            ),
            'cn': self.get_stroke_data(
                f'{get_package_dir()}/data/handwriting-zh_CN.xml'
            ),
            'cn_Hant': self.get_stroke_data(
                f'{get_package_dir()}/data/handwriting-zh_TW.xml'
            )
        }

    def len_keys(self):
        L = []
        for k in self.D:
            L.extend(list(self.D[k][0].keys()))
        return sorted(self.D.keys())

    def iter_len(self, len_, LKeys=None):
        D = self.D
        if LKeys is None:
            LKeys = list(D.keys())

        for key in LKeys:
            if not len_ in D[key][0]:
                continue

            for ord_, LStrokes in D[key][0][len_]:
                yield key, ord_, LStrokes

    def iter(self, LKeys=None):
        if LKeys is None:
            LKeys = list(self.D.keys())

        LOrds = []
        for k in LKeys:
            LOrds.extend(list(self.D[k][1].keys()))

        for ord_ in sorted(LOrds):
            yield ord_, self.get(
                ord_, LKeys=LKeys
            )

    def get(self, ord_, LKeys=None):
        if LKeys is None:
            LKeys = list(self.D.keys())

        L = []
        for k in LKeys:
            if ord_ in self.D[k][1]:
                L.append(self.D[k][1][ord_])
        return L

    def get_stroke_data(self, path):
        DLCandLens = {}
        DCands = {}

        for ord_, LStrokes in iter_tomoe_data(path):
            # Add stroke angle etc information
            #print ord_, LStrokes
            LStrokes = stroke_metrics.get_L_strokes(LStrokes)

            DLCandLens.setdefault(len(LStrokes), []).append(
                (ord_, LStrokes)
            )
            DCands[ord_] = LStrokes

        return DLCandLens, DCands

