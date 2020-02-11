import json
import numpy as np
import tensorflow as tf
from cnn_chinese_hw.get_package_dir import get_package_dir
from cnn_chinese_hw.recognizer.TomoeDataset import TomoeDataset


if __name__ == '__main__':
    # Quantize to file (reduce amount of memory/disk space needed)
    model = tf.keras.models.load_model(f'{get_package_dir()}/data/hw_model.hdf5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_quant_model = converter.convert()

    with open(f'{get_package_dir()}/data/hw_quant_model.tflite', 'wb') as f:
        f.write(tflite_quant_model)

    # Output class names to file
    with open(f'{get_package_dir()}/data/strokedata.npz', 'rb') as f:
        files = np.load(f)
        class_names = files['class_names']

    with open(f'{get_package_dir()}/data/hw_quant_classes.json', 'w') as f:
        f.write(json.dumps([int(i) for i in class_names]))
