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
            'zh': self.get_stroke_data(
                f'{get_package_dir()}/data/handwriting-zh_CN.xml'
            ),
            'zh-Hant': self.get_stroke_data(
                f'{get_package_dir()}/data/handwriting-zh_TW.xml'
            )
        }

    def len_keys(self):
        L = []
        for k in self.D:
            L.extend(list(self.D[k][0].keys()))
        return sorted(self.D.keys())

    def iter_len(self, len_, keys_list=None):
        D = self.D
        if keys_list is None:
            keys_list = list(D.keys())

        for key in keys_list:
            if not len_ in D[key][0]:
                continue

            for ord_, strokes_list in D[key][0][len_]:
                yield key, ord_, strokes_list

    def iter(self, keys_list=None):
        if keys_list is None:
            keys_list = list(self.D.keys())

        LOrds = []
        for k in keys_list:
            LOrds.extend(list(self.D[k][1].keys()))

        for ord_ in sorted(LOrds):
            yield ord_, self.get(
                ord_, keys_list=keys_list
            )

    def get(self, ord_, keys_list=None):
        if keys_list is None:
            keys_list = list(self.D.keys())

        L = []
        for k in keys_list:
            if ord_ in self.D[k][1]:
                L.append(self.D[k][1][ord_])
        return L

    def get_stroke_data(self, path):
        DLCandLens = {}
        DCands = {}

        for ord_, strokes_list in iter_tomoe_data(path):
            # Add stroke angle etc information
            #print ord_, strokes_list
            strokes_list = stroke_metrics.get_L_strokes(strokes_list)

            DLCandLens.setdefault(len(strokes_list), []).append(
                (ord_, strokes_list)
            )
            DCands[ord_] = strokes_list

        return DLCandLens, DCands

