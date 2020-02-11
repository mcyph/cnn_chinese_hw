from cnn_chinese_hw.recognizer.TFLiteRecognizer import TFLiteRecognizer
from cnn_chinese_hw.client_server.rem_dupes import rem_dupes

from speedysvc.rpc_decorators import json_method
from speedysvc.client_server.base_classes.ServerMethodsBase import ServerMethodsBase


USE_ZINNIA = False


class HWServer(ServerMethodsBase):
    port = 40511
    name = 'hw'

    def __init__(self, logger_client):
        self.recognizer = TFLiteRecognizer()
        ServerMethodsBase.__init__(self, logger_client)

    @json_method
    def get_cn_written_cand(self, LStrokes, id):
        if not LStrokes:
            # TODO: Also handle single stroke candidates differently!
            return {'LCands': [], 'id': id}

        LRtn = [
            chr(ord_) for _, ord_ in
            self.recognizer.get_L_candidates(LStrokes)
        ]

        LRtn = rem_dupes(LRtn)
        DRtn = {'LCands': LRtn, 'id': id}
        return DRtn
