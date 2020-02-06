import os
import numpy as np

from cnn_chinese_hw.parse_data.StrokeData import StrokeData
from cnn_chinese_hw.stroke_tools.HWDataAugmenter import HWStrokesAugmenter
from cnn_chinese_hw.gui.HandwritingRegister import HandwritingRegister
from cnn_chinese_hw.stroke_tools.get_vertex import get_vertex
from cnn_chinese_hw.stroke_tools.points_normalized import points_normalized


def shuffle_in_unison_inplace(a, b):
    """
    Shuffle two arrays together
    @param a: numpy array
    @param b: numpy array
    @return: two numpy arrays shuffled
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


class TomoeDataset:
    def __init__(self,
                 image_size,
                 augmentations_per_sample,
                 small_sample_only,
                 small_sample_size,
                 num_test,
                 load_images=True,
                 cache=True):

        self.image_size = image_size
        self.augmentations_per_sample = augmentations_per_sample
        self.small_sample_only = small_sample_only
        self.small_sample_size = small_sample_size
        self.num_test = num_test
        self.load_images = load_images

        if small_sample_only:
            self.dataset_path = 'strokedata_sample.npz'
        else:
            self.dataset_path = 'strokedata.npz'

        if cache and os.path.exists(self.dataset_path):
            self.__load_from_file(self.dataset_path)
        else:
            self.__init_data()
            self.__save_to_file(self.dataset_path)

    def __load_from_file(self, path):
        with open(path, 'rb') as f:
            files = np.load(f)

            self.train_images = files['train_images'] if self.load_images else None
            self.train_labels = files['train_labels']
            self.test_images = files['test_images']
            self.test_labels = files['test_labels']
            self.class_names = files['class_names']

    def __save_to_file(self, path):
        with open(path, 'wb') as f:
            np.savez(
                f,
                train_images=self.train_images,
                train_labels=self.train_labels,
                test_images=self.test_images,
                test_labels=self.test_labels,
                class_names=self.class_names
            )

    def __init_data(self):
        LClassOrds = []
        DRasteredByOrd = {}
        DOrdToClass = {}
        cur_class_idx = 0
        num_images = 0

        for ord_, rastered in self.iter_data_labels():
            if not ord_ in DRasteredByOrd:
                DOrdToClass[ord_] = cur_class_idx
                LClassOrds.append(ord_)
                cur_class_idx += 1
                DRasteredByOrd[ord_] = []

            DRasteredByOrd[ord_].append(rastered)
            num_images += 1

        # not-float64 bottom-out warning!
        data = np.zeros((num_images, self.image_size, self.image_size),
                        dtype=np.float32)
        labels = np.zeros(num_images, dtype=np.uint32)

        xx = 0
        for ord_, LRastered in DRasteredByOrd.items():
            for rastered in LRastered:
                data[xx, :, :] = rastered
                labels[xx] = DOrdToClass[ord_]
                xx += 1

                if xx % 1000 == 0:
                    print("Data #:", xx)

        data, labels = shuffle_in_unison_inplace(data, labels)
        data = data / 255.0

        self.train_images = data[self.num_test:]
        self.train_labels = labels[self.num_test:]
        self.test_images = data[:self.num_test]
        self.test_labels = labels[:self.num_test]
        self.class_names = np.asarray(LClassOrds, dtype='int32')

    def iter_data_labels(self):
        # Add from my data
        hr = HandwritingRegister()
        DStrokes = hr.get_D_strokes()
        for ord_, LStrokes in DStrokes.items():
            print("Processing my data:", ord_, chr(ord_))
            for i_LStrokes in LStrokes:
                i_LStrokes = points_normalized(i_LStrokes)
                i_LStrokes = [get_vertex(i) for i in i_LStrokes]

                aug = HWStrokesAugmenter(i_LStrokes)

                for __ in range(self.augmentations_per_sample):
                    # Render on top lots of times, to give an idea
                    # of where the lines will end up on average.
                    rastered = aug.raster_strokes(on_val=255, image_size=self.image_size)
                    yield ord_, rastered

        # Add from Tomoe data
        sd = StrokeData()
        SAlwaysAdd = set(
            ord(i) for i in '目木日書的一是不了人我在有他这为之大来以个中上们'
        )
        for xx, (ord_, LStrokes) in enumerate(sd.iter()):
            if ord_ in SAlwaysAdd:
                pass
            elif self.small_sample_only and xx > self.small_sample_size:
                continue
            print("Processing Tomoe data:", ord_, chr(ord_))

            for i_LStrokes in LStrokes:
                strokes = [i.LPoints for i in i_LStrokes]
                aug = HWStrokesAugmenter(strokes)
                for __ in range(self.augmentations_per_sample):
                    rastered = aug.raster_strokes(on_val=255, image_size=self.image_size)
                    yield ord_, rastered


if __name__ == '__main__':
    pass
