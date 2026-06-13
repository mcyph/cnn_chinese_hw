from cnn_chinese_hw.gui.HandwritingCanvas import HandwritingCanvas


class HandwritingRegisterCanvas(HandwritingCanvas):
    def __init__(self, parent, id=-1, size=(1000, 1000)):
        HandwritingCanvas.__init__(self, parent, id, size)

    def process_lines(self, LVertices):
        pass
