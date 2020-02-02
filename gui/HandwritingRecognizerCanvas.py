from cnn_chinese_hw.recognizer import HandwritingModel
from cnn_chinese_hw.gui.HandwritingCanvas import HandwritingCanvas
from cnn_chinese_hw.stroke_tools.HWDataAugmenter import HWStrokesAugmenter


class HandwritingRecogniserCanvas(HandwritingCanvas):
    def __init__(self, parent, id=-1, size=(1000,1000)):
        HandwritingModel.CACHE_MODEL = True
        HandwritingModel.CACHE_DATASET = True
        HandwritingModel.SMALL_SAMPLE_ONLY = False
        HandwritingModel.LCHECK_ORD = 0
        self.hw_model = HandwritingModel.HandwritingModel(load_images=False)
        self.hw_model.run()

        HandwritingCanvas.__init__(self, parent, id, size)

    def process_lines(self, LVertices):
        # Get the rastered character (normal)
        aug = HWStrokesAugmenter(LVertices)
        rastered = aug.raster_strokes(on_val=255,
                                      image_size=HandwritingModel.IMAGE_SIZE,
                                      do_augment=False) / 255.0

        # Get the rastered character (randomly rotated, etc)
        LAugRastered = [
            aug.raster_strokes(on_val=255,
                               image_size=HandwritingModel.IMAGE_SIZE) / 255.0
            for ___ in range(3)
        ]

        # Show the predictions
        self.hw_model.do_prediction(rastered,
                                    should_be_ord=0,
                                    LAugRastered=LAugRastered)

