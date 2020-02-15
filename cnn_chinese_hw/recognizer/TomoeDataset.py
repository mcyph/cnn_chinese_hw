import os
import json
from hashlib import md5
import numpy as np

from cnn_chinese_hw.parse_data.iter_kanjivg_data import iter_kanjivg_data
from cnn_chinese_hw.parse_data.StrokeData import StrokeData
from cnn_chinese_hw.stroke_tools.HWDataAugmenter import HWStrokesAugmenter
from cnn_chinese_hw.gui.HandwritingRegister import HandwritingRegister
from cnn_chinese_hw.stroke_tools.get_vertex import get_vertex
from cnn_chinese_hw.stroke_tools.points_normalized import points_normalized
from cnn_chinese_hw.get_package_dir import get_package_dir


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
                 real_strokes_per_sample_times,
                 small_sample_only,
                 small_sample_size,
                 load_images=True,
                 cache=True):

        self.image_size = image_size
        self.augmentations_per_sample = augmentations_per_sample
        self.real_strokes_per_sample_times = real_strokes_per_sample_times
        self.small_sample_only = small_sample_only
        self.small_sample_size = small_sample_size
        self.load_images = load_images

        self.DOrdToClass = {}
        self.LClassOrds = []

        if small_sample_only:
            self.dataset_path = f'{get_package_dir()}/data/' \
                                f'strokedata_sample.npz'
        else:
            self.dataset_path = f'{get_package_dir()}/data/' \
                                f'strokedata.npz'

        if cache and os.path.exists(self.dataset_path):
            self.__load_from_file(self.dataset_path)
        else:
            self.__init_data()
            self.__save_to_file(self.dataset_path)

    #=========================================================#
    #                  Read/Write from Cache                  #
    #=========================================================#

    def __load_from_file(self, path):
        with open(path, 'rb') as f:
            print("Loading Tomoe Dataset from file:", path)
            files = np.load(f)

            self.train_images = (
                files['train_images']
                if self.load_images
                else None
            )
            self.train_labels = files['train_labels']
            self.test_images = files['test_images']
            self.test_labels = files['test_labels']
            self.class_names = files['class_names']

    def __save_to_file(self, path):
        with open(path, 'wb') as f:
            print("Saving Tomoe Dataset to file:", path)
            np.savez(
                f,
                train_images=self.train_images,
                train_labels=self.train_labels,
                test_images=self.test_images,
                test_labels=self.test_labels,
                class_names=self.class_names
            )

    def __init_data(self):
        train_images, train_labels = self.__get_data(self.__iter_tomoe_data_labels)
        train_images, train_labels = shuffle_in_unison_inplace(train_images, train_labels)
        self.train_images = train_images
        self.train_labels = train_labels

        test_images, test_labels = self.__get_data(self.__iter_kanjivg_data_labels)
        test_images, test_labels = shuffle_in_unison_inplace(test_images, test_labels)
        self.test_images = test_images
        self.test_labels = test_labels

        self.class_names = np.asarray(
            self.LClassOrds, dtype='int32'
        )

    def __get_data(self, fn):
        DRasteredByOrd = {}
        cur_class_idx = 0
        num_images = 0

        for ord_, rastered in fn():
            if not ord_ in self.DOrdToClass:
                self.DOrdToClass[ord_] = cur_class_idx
                self.LClassOrds.append(ord_)
                cur_class_idx += 1

            if not ord_ in DRasteredByOrd:
                DRasteredByOrd[ord_] = []

            DRasteredByOrd[ord_].append(rastered)
            num_images += 1

        # not-float64 bottom-out warning!
        data = np.zeros((num_images,
                         self.image_size,
                         self.image_size,
                         2),
                        dtype=np.uint8)
        labels = np.zeros(num_images, dtype=np.uint32)

        xx = 0
        for ord_ in list(DRasteredByOrd.keys()):
            LRastered = DRasteredByOrd[ord_]
            del DRasteredByOrd[ord_]  # save memory

            for rastered in LRastered:
                # Again am using uint8 to save memory during generation
                data[xx, :, :, :] = rastered
                labels[xx] = self.DOrdToClass[ord_]
                xx += 1

                if xx % 1000 == 0:
                    print("Data #:", xx)

        return data, labels

    def __iter_kanjivg_data_labels(self):
        """
        Get from KanjiVG for validation
        """
        # TODO: Should this test on the whole sample or a small selection?
        #   If using a small selection, will there be consequences if/if
        #   not samples are randomly selected?

        for ord_, LStrokes in iter_kanjivg_data():
            print("Processing KanjiVG data:", ord_, chr(ord_))
            aug = HWStrokesAugmenter(LStrokes, find_vertices=True)
            rastered = aug.raster_strokes(
                image_size=self.image_size,
                do_augment=False
            )
            yield ord_, rastered

    def __iter_tomoe_data_labels(self):
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
                    rastered = aug.raster_strokes(
                        image_size=self.image_size
                    )
                    yield ord_, rastered

                for __ in range(self.real_strokes_per_sample_times):
                    rastered = aug.raster_strokes(
                        image_size=self.image_size,
                        do_augment=False
                    )
                    yield ord_, rastered

        # Add from Tomoe data
        sd = StrokeData()
        SAlwaysAdd = set(
            ord(i) for i in '目木日書的一是不了人我在有他这为之大来以个中上们'
        )
        SAdded = set()

        for xx, (ord_, LStrokes) in enumerate(sd.iter()):
            if ord_ in SAlwaysAdd:
                pass
            elif self.small_sample_only and xx > self.small_sample_size:
                continue
            print("Processing Tomoe data:", ord_, chr(ord_))

            # Remove duplicate data from Chinese Simplified+Traditional/Japanese
            m = md5()
            m.update(chr(ord_).encode('utf-8'))
            m.update(json.dumps(LStrokes).encode('ascii'))
            digest = m.digest()
            if digest in SAdded:
                print("Ignoring dupe data:", ord_, chr(ord_))
                continue
            SAdded.add(digest)

            for i_LStrokes in LStrokes:
                strokes = [i.LPoints for i in i_LStrokes]
                aug = HWStrokesAugmenter(strokes)
                for __ in range(self.augmentations_per_sample):
                    rastered = aug.raster_strokes(
                        image_size=self.image_size
                    )
                    yield ord_, rastered

                for __ in range(self.real_strokes_per_sample_times):
                    rastered = aug.raster_strokes(
                        image_size=self.image_size,
                        do_augment=False
                    )
                    yield ord_, rastered


if __name__ == '__main__':
    pass
