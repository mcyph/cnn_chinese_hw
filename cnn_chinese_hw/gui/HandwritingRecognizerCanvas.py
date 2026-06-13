from cnn_chinese_hw.gui.HandwritingCanvas import HandwritingCanvas
from cnn_chinese_hw.recognizer.recognizer import HandwritingRecognizer


class HandwritingRecogniserCanvas(HandwritingCanvas):
    def __init__(self, parent, id=-1, size=(1000, 1000)):
        self.recognizer = HandwritingRecognizer()
        HandwritingCanvas.__init__(self, parent, id, size)

    def process_lines(self, LVertices):
        for score, ord_ in self.recognizer.get_candidates_list(LVertices, n_cands=10):
            print(f"prediction: {score:.4f} {chr(ord_)}")
