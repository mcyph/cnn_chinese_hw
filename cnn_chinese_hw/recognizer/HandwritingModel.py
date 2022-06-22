from cnn_chinese_hw.stroke_tools.HWDataAugmenter import HWStrokesAugmenter

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from cnn_chinese_hw.recognizer.TomoeDataset import TomoeDataset
from cnn_chinese_hw.get_package_dir import get_package_dir


"""
The parameters I tried here that made the 
most difference to the validation loss were:

Changing dropout -> BatchNormalization: 2.4+ -> 2.01
Changing from 50 augmented 
    -> 50 aug and 10 copies of real strokes: -> 1.86
Increase the random augmenter constants: -> 1.74 
"""


# use less memory than float32
# OPEN ISSUE: Would it be better to support float16 here?
keras.backend.set_floatx('float32')


NUM_EPOCHS = 1000
# 32 -> 2.52 validation loss
# 128 -> 2.04
# 256 -> 2.01
# 1024 -> 2.016
# 512 is most I can fit into memory for 2 channel
BATCH_SIZE = 384

IMAGE_SIZE = 28
CHANNELS = 3
# How many times to augment each set of strokes
# (i.e. rotate/scale/distort... etc)
# 150 needs 8.69 GiB - want to be sure
# I don't run out of RAM
AUGMENTATIONS_PER_SAMPLE = 40
# How often to add the actual
# (unmodified) strokes
REAL_STROKES_PER_SAMPLE_TIMES = 10
CACHE_DATASET = True
CACHE_MODEL = False

# For testing
SMALL_SAMPLE_ONLY = False
SMALL_SAMPLE_SIZE = 500


class HandwritingModel:
    def __init__(self, load_images=True):
        # Cache the Kanji data if possible,
        # as can take quite a long time itself
        if SMALL_SAMPLE_ONLY:
            self.model_path = f'{get_package_dir()}/data/hw_model_sample.hdf5'
        else:
            self.model_path = f'{get_package_dir()}/data/hw_model.hdf5'

        self.dataset = TomoeDataset(
            image_size=IMAGE_SIZE,
            augmentations_per_sample=AUGMENTATIONS_PER_SAMPLE,
            real_strokes_per_sample_times=REAL_STROKES_PER_SAMPLE_TIMES,
            small_sample_only=SMALL_SAMPLE_ONLY,
            small_sample_size=SMALL_SAMPLE_SIZE,
            load_images=load_images,
            cache=CACHE_DATASET
        )

        xx = 0
        for x, label in enumerate(self.dataset.train_labels):
            #print("LABEL:", label, self.dataset.class_names[label], LCHECK_ORD)
            if self.dataset.class_names[label] == LCHECK_ORD:
                print("FOUND!!!")
                img = self.dataset.train_images[x]
                self.matshow(img[:, :, 0], img[:, :, 1], img[:, :, 2])

                xx += 1
                if xx > 10:
                    break

    def matshow(self, ch1, ch2=None, ch3=None):
        three_chans = np.zeros(shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                               dtype='uint8')
        three_chans[:, :, 0] = ch1
        if ch2 is not None:
            three_chans[:, :, 1] = ch2
        if ch3 is not None:
            three_chans[:, :, 2] = ch3
        plt.matshow(three_chans,
                    interpolation='nearest',
                    vmin=0.0,
                    vmax=1.0)
        plt.show()

    def run(self):
        if CACHE_MODEL:
            self.model = keras.models.load_model(self.model_path)
        else:
            self.model = self.cnn_model()

    def cnn_model(self):
        x_train, y_train = self.dataset.train_images, \
                           self.dataset.train_labels
        x_train = x_train.astype('float32').reshape(
            -1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS
        ) / 255.0

        x_val, y_val = self.dataset.test_images, \
                       self.dataset.test_labels
        x_val = x_val.astype('float32').reshape(
            -1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS
        ) / 255.0
        print("NUM TRAIN VALUES:", x_val.shape)

        # Various resources I used in coming to these parameters:
        # https://github.com/jtyoui/Jtyoui/blob/master/jtyoui/neuralNetwork/kerase/HandWritingRecognition.py (MIT)
        # https://pdfs.semanticscholar.org/4941/aed85462968e9918110b4ba740c56030fd23.pdf
        # "Hands-On Machine Learning with Scikit-Learn and TensorFlow" by Aurelien Geron
        # https://www.kdnuggets.com/2018/09/dropout-convolutional-networks.html
        # https://towardsdatascience.com/deep-study-of-a-not-very-deep-neural-network-part-2-activation-functions-fd9bd8d406fc
        # https://missinglink.ai/guides/keras/keras-conv2d-working-cnn-2d-convolutions-keras/

        l2 = keras.regularizers.l2
        # If this value is too high, it will
        #    underfit and be slow to converge.
        # If it's too low, it will be allowed to overfit.
        # I suspect the best value might be between 0.01-0.001
        # 0.01 definitely underfits, not sure if 0.005 is too low/high.
        # Suspect closer to 0.001 might be a good value.
        l2_l = 0.0005

        def conv2d(filters):
            return keras.layers.Convolution2D(
                filters=filters,
                kernel_size=3,
                padding='same',
                activation='elu',
                kernel_regularizer=l2(l2_l)
            )

        def dense(units):
            return keras.layers.Dense(
                units=units,
                activation='elu',
                kernel_regularizer=l2(l2_l)
            )

        model = self.model = keras.Sequential([
            keras.layers.Convolution2D(input_shape=(IMAGE_SIZE,
                                                    IMAGE_SIZE,
                                                    CHANNELS),
                                       filters=64,  # Number of outputs
                                       # Might be a good idea to try 5,
                                       # only for the first layer?
                                       kernel_size=3,
                                       strides=1,
                                       padding='same',
                                       activation='elu',
                                       kernel_regularizer=l2(l2_l)),
            keras.layers.BatchNormalization(),
            conv2d(64),
            keras.layers.BatchNormalization(),
            conv2d(64),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=1,
                                   strides=2,
                                   padding='same'),

            conv2d(128),
            keras.layers.BatchNormalization(),
            conv2d(128),
            keras.layers.BatchNormalization(),
            #conv2d(128),
            #keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=1,
                                   strides=2,
                                   padding='same'),

            conv2d(256),
            keras.layers.BatchNormalization(),
            conv2d(256),
            keras.layers.BatchNormalization(),
            #conv2d(256),
            #keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=1,
                                   strides=2,
                                   padding='same'),

            keras.layers.Flatten(),

            dense(2048),
            keras.layers.BatchNormalization(),
            dense(1024),
            keras.layers.BatchNormalization(),
            #dense(1024),
            #keras.layers.BatchNormalization(),
            dense(1024),

            # Perhaps this is not the right place for this?
            # https://stats.stackexchange.com/questions/299292/dropout-makes-performance-worse
            #keras.layers.Dropout(0.5),

            keras.layers.Dense(units=len(self.dataset.class_names),
                               activation=keras.activations.softmax,
                               kernel_regularizer=l2(l2_l))
        ])
        # Save on memory
        del self.dataset

        # Not sure if Adam or SGD is better here.
        # Suspect SGD might be slower to converge,
        # but give better generalization.
        opt = keras.optimizers.Adam(lr=1e-5)
        # learning_rate of 0.01 might be best when testing?
        #opt = keras.optimizers.SGD(learning_rate=0.01, nesterov=True)
        model.compile(
            optimizer=opt,
            #loss=keras.losses.categorical_crossentropy,
            loss=keras.losses.sparse_categorical_crossentropy,
            metrics=[
                'accuracy',
                #'mae'
            ]
        )

        # Suspect val_accuracy might be more important than
        # val_loss, as the correct result should be the first
        # one the majority of the time (seeing as it's a simple
        # binary "was it correctly predicted"). Good idea to
        # make sure val_loss doesn't get too high by a
        # significant amount, though
        #
        # Interesting article:
        # http://alexadam.ca/ml/2018/08/03/early-stopping.html

        es = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            verbose=1,
            patience=15,
            #min_delta=1
        )
        mc = keras.callbacks.ModelCheckpoint(
            self.model_path,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True
        )
        model.fit(
            x=x_train,
            y=y_train,
            batch_size=BATCH_SIZE,
            epochs=NUM_EPOCHS,
            validation_data=(x_val, y_val),
            callbacks=[es, mc]
        )
        return model

    def do_prediction(self, rastered, should_be_ord, LAugRastered=None):
        # Convert it to a format the model understands
        rastered = rastered.reshape(
            -1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS
        ).astype('float32') / 255.0
        #rastered = rastered.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
        print("RASTERED:", rastered)

        self.matshow(
            rastered[0][:, :, 0]*255,
            rastered[0][:, :, 1]*255,
            rastered[0][:, :, 2]*255
        )

        # Output the best prediction
        predictions = self.model.predict(rastered)
        assert len(predictions) == 1
        predictions = predictions[0]

        LPredict = []
        for idx, prediction in enumerate(predictions):
            if prediction > 0:
                LPredict.append((prediction, idx))

            if should_be_ord == self.dataset.class_names[idx]:
                print("CORRECT MATCH->SHOULD BE HIGH:", prediction)
        LPredict.sort(key=lambda x: -x[0])

        for xx, (prediction, idx) in enumerate(LPredict):
            # print(prediction)
            print("PREDICTION:",
                  prediction, idx,
                  self.dataset.class_names[idx],
                  chr(self.dataset.class_names[idx]))
            if xx > 10:
                break

        if LAugRastered:
            result = np.zeros(shape=(len(self.dataset.class_names),),
                              dtype='float32')
            for i_rastered in LAugRastered:
                i_rastered = i_rastered.reshape(
                    -1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS
                ).astype('float32') / 255.0
                result += self.model.predict(i_rastered)[0]

            LPredict = []
            for idx, prediction in enumerate(result):
                if prediction > 0:
                    LPredict.append((prediction, idx))

                if should_be_ord == self.dataset.class_names[idx]:
                    print("AUG CORRECT MATCH->SHOULD BE HIGH:", prediction)
            LPredict.sort(key=lambda x: -x[0])

            for xx, (prediction, idx) in enumerate(LPredict):
                # print(prediction)
                print("AUG PREDICTION:",
                      prediction, idx,
                      self.dataset.class_names[idx],
                      chr(self.dataset.class_names[idx]))
                if xx > 10:
                    break


if __name__ == '__main__':
    # Define a test handwritten thing
    LCHECK = [[(208, 0), (199, 119), (94, 341)],
              [(0, 461), (781, 520), (915, 520), (999, 479)],
              [(189, 167), (213, 209), (238, 826), (268, 934), (203, 910)],
              [(303, 514), (94, 766)],
              [(462, 17), (497, 586), (522, 688), (646, 886), (796, 1000)],
              [(716, 628), (462, 916)],
              [(696, 101), (771, 155), (835, 251)]]
    LCHECK_ORD = ord('我')

    aug = HWStrokesAugmenter(LCHECK,
                             find_vertices=True)
    LCHECK_RASTERED = aug.raster_strokes(image_size=IMAGE_SIZE,
                                         do_augment=False).astype('float32') / 255.0
    plt.matshow(LCHECK_RASTERED)
    plt.show()

    LCHECK_RASTERED_AUG = [
        aug.raster_strokes(image_size=IMAGE_SIZE).astype('float32') / 255.0
        for ___ in range(20)
    ]

    hw_model = HandwritingModel()
    hw_model.run()
    #hw_model.do_prediction(LCHECK_RASTERED, LCHECK_ORD,
    #                   LAugRastered=LCHECK_RASTERED_AUG)

    #for x, img in enumerate(demo.dataset.train_images):
    #    should_be_ord = hw_model.dataset.class_names[hw_model.dataset.train_labels[x]]
    #    print('ENUMERATE x:', x, hw_model.dataset.train_labels[x],
    #          chr(should_be_ord))
    #    hw_model.do_prediction(img, should_be_ord)
    #    if x > 30:
    #        break
