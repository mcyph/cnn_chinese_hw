from json import loads, dumps

from cnn_chinese_hw.parse_data import StrokeData
from cnn_chinese_hw.recognizer import Recognizer

from input.toolkit.list_operations.rem_dupes import rem_dupes

from speedyrpc.rpc_decorators import json_method
from speedyrpc.client_server.base_classes.ServerMethodsBase import ServerMethodsBase


USE_ZINNIA = False


class HWServer(ServerMethodsBase):
    port = 40511
    name = 'hw'

    def __init__(self, logger_client):
        print('Loading stroke data...')
        self.stroke_data = stroke_data = StrokeData()
        print('Creating slow old recognizer instance...')
        self.recognizer = Recognizer(stroke_data)
        if USE_ZINNIA:
            print('Creating zinnia recognizer instance...')
            from cnn_chinese_hw.recognizer.ZinniaRecognizer import ZinniaRecognizer
            self.zinnia_recognizer = ZinniaRecognizer()
        print('OK!')

        ServerMethodsBase.__init__(self, logger_client)

    @json_method
    def get_cn_written_cand(self, LStrokes, id, DOpts):

        if not LStrokes:
            # HACK!
            return dumps({'LCands': LStrokes, 'id': id})

        #DOpts = json_tools.loads(DOpts)

        # LPoints = []
        # for LStroke in LStrokes:
        #    LStroke = points_to_plot([LStroke])[0]
        #    if not LStroke: continue
        #
        #    Line1 = get_vertex(LStroke)
        #    LPoints.append(Line1)
        # LPoints = points_normalized(LPoints)

        print('Recognizing:', LStrokes)

        #DMap = {
        #    'CnUnified': ['cn', 'ja'],
        #    'CnSimp': ['cn'],
        #    'CnTrad': ['ja'],  # FIXME!
        #    'Japanese': ['ja']
        #}

        #t = Timer()
        if USE_ZINNIA:
            LRtn = [
                chr_ for _, chr_ in
                self.zinnia_recognizer.recognize_out_of_order(
                    'ja', # HACK!!! ==========================================================================
                    LStrokes
                )
            ]
        else:
            LRtn = [
                chr(ord_) for _, ord_ in
                self.recognizer.get_L_cands(
                    LStrokes, DOpts
                )[::-1]
            ]

        LRtn = rem_dupes(LRtn)
        #print('CnWrittenCand Get Candidates:', t)
        DRtn = {'LCands': LRtn, 'id': id}
        return DRtn


if __name__ == '__main__':
    server = HWServer()
    while 1:
        from time import sleep
        sleep(10)
