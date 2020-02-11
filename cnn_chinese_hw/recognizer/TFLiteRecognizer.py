# Recognize using the model quantized for tflite

import json
import _thread
import numpy as np
try:
    import tensorflow
    Interpreter = tensorflow.lite.Interpreter
except ImportError:
    import tflite_runtime.interpreter
    Interpreter = tflite_runtime.interpreter.Interpreter

# HACK: This should be moved from HandwritingModel
# to remove dependency on (non-lite) tensorflow!
IMAGE_SIZE = 28

from cnn_chinese_hw.get_package_dir import get_package_dir
from cnn_chinese_hw.stroke_tools.HWDataAugmenter import HWStrokesAugmenter


class TFLiteRecognizer:
    def __init__(self):
        """

        """
        # Load TFLite model and allocate tensors.
        interpreter = self.interpreter = Interpreter(
            model_path=f'{get_package_dir()}/data/hw_quant_model.tflite'
        )
        interpreter.allocate_tensors()

        with open(f'{get_package_dir()}/data/'
                  f'hw_quant_classes.json', 'r') as f:
            self.classnames = json.loads(f.read())

        # Get input and output tensors.
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()

        self.lock = _thread.allocate_lock()

    def get_L_candidates(self, LStrokes, n_cands=20):
        """
        Get Kanji/Hanzi handwriting candidates

        :param LStrokes: a list of strokes in x,y point format
                         [[(x, y), ...], ...]
        :param n_cands: number of candidates to return
        :return: a list of [(candidate score, candidate ordinal), ...]
                 (to convert the ordinal to a character, use chr(ordinal))
        """

        # Get the rastered character (normal)
        aug = HWStrokesAugmenter(LStrokes)
        rastered = aug.raster_strokes(
            on_val=255, image_size=IMAGE_SIZE, do_augment=False
        ) / 255.0
        rastered = rastered.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
        rastered = rastered.astype('float32')

        with self.lock:
            self.interpreter.set_tensor(
                self.input_details[0]['index'], rastered
            )
            self.interpreter.invoke()

            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            output_data = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )[0]

        idx = output_data.argsort()[-n_cands:][::-1]
        return [
            (output_data[i], self.classnames[i])
            for i in idx
        ]


if __name__ == '__main__':
    recognizer = TFLiteRecognizer()
    for x in range(10000):
        print(recognizer.get_L_candidates([]))
