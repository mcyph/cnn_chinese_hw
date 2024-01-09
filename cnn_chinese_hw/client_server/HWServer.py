from cnn_chinese_hw.client_server.rem_dupes import rem_dupes
from cnn_chinese_hw.recognizer.TFLiteRecognizer import TFLiteRecognizer


USE_ZINNIA = False


class HWServer:
    def __init__(self):
        self.recognizer = TFLiteRecognizer()

    def get_cn_written_cand(self, strokes_list, id):
        if not strokes_list:
            # TODO: Also handle single stroke candidates differently!
            return {'cands_list': [], 'id': id}

        return_list = [chr(ord_)
                       for _, ord_
                       in self.recognizer.get_candidates_list(strokes_list)]

        return_list = rem_dupes(return_list)
        return_dict = {'cands_list': return_list, 'id': id}
        return return_dict
