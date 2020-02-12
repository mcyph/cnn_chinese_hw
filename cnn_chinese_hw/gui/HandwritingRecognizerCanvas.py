from cnn_chinese_hw.recognizer import HandwritingModel
from cnn_chinese_hw.gui.HandwritingCanvas import HandwritingCanvas
from cnn_chinese_hw.stroke_tools.HWDataAugmenter import HWStrokesAugmenter
from cnn_chinese_hw.recognizer.TFLiteRecognizer import TFLiteRecognizer


class HandwritingRecogniserCanvas(HandwritingCanvas):
    def __init__(self, parent, id=-1, size=(1000,1000)):
        HandwritingModel.CACHE_MODEL = True
        HandwritingModel.CACHE_DATASET = True
        HandwritingModel.SMALL_SAMPLE_ONLY = False
        HandwritingModel.LCHECK_ORD = 0
        self.hw_model = HandwritingModel.HandwritingModel(load_images=False)
        self.hw_model.run()

        self.tflite_recognizer = TFLiteRecognizer()

        HandwritingCanvas.__init__(self, parent, id, size)

    def process_lines(self, LVertices):
        # Get the rastered character (normal)
        aug = HWStrokesAugmenter(LVertices)
        rastered = aug.raster_strokes(image_size=HandwritingModel.IMAGE_SIZE,
                                      do_augment=False) / 255.0

        # Get the rastered character (randomly rotated, etc)
        LAugRastered = [
            aug.raster_strokes(image_size=HandwritingModel.IMAGE_SIZE) / 255.0
            for ___ in range(3)
        ]

        # Show the predictions using the
        # tensorflow lite quantized model
        self.hw_model.do_prediction(rastered,
                                    should_be_ord=0,
                                    LAugRastered=LAugRastered)

        for score, ord_ in self.tflite_recognizer.get_L_candidates(LVertices, n_cands=10):
            print(f"tflite prediction: {score} {chr(ord_)}")
