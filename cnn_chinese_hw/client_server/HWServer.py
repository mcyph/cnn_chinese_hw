from cnn_chinese_hw.client_server.rem_dupes import rem_dupes
from cnn_chinese_hw.recognizer.recognizer import HandwritingRecognizer


class HWServer:
    """Thin service wrapper around :class:`HandwritingRecognizer`.

    The recognizer (and its checkpoint) is loaded lazily on first use, so
    importing this module never requires a trained model to be present -- the
    error surfaces only when a recognition request is actually made.
    """

    def __init__(self, device='cpu'):
        self._device = device
        self._recognizer = None

    @property
    def recognizer(self):
        if self._recognizer is None:
            self._recognizer = HandwritingRecognizer(device=self._device)
        return self._recognizer

    def get_chinese_written_candidates(self, strokes_list, id):
        if not strokes_list:
            # TODO: Also handle single stroke candidates differently!
            return {'cands_list': [], 'id': id}

        return_list = [chr(ord_)
                       for _, ord_
                       in self.recognizer.get_candidates_list(strokes_list)]

        return_list = rem_dupes(return_list)
        return {'cands_list': return_list, 'id': id}
