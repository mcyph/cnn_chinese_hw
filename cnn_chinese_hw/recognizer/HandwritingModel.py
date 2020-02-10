from cnn_chinese_hw.stroke_tools.HWDataAugmenter import HWStrokesAugmenter

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from cnn_chinese_hw.recognizer.TomoeDataset import TomoeDataset
from cnn_chinese_hw.get_package_dir import get_package_dir

# use less memory than float32
# OPEN ISSUE: Would it be better to support float16 here?
keras.backend.set_floatx('float32')


NUM_EPOCHS = 10
BATCH_SIZE = 256

IMAGE_SIZE = 28
# For validation of model
NUM_TEST = 0
# 150 needs 8.69 GiB - want to be sure
# my 4GB card won't run out of memory
AUGMENTATIONS_PER_SAMPLE = 50
OVERLAY_RANDOM = False
CACHE_DATASET = False
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
            small_sample_only=SMALL_SAMPLE_ONLY,
            small_sample_size=SMALL_SAMPLE_SIZE,
            num_test=NUM_TEST,
            load_images=load_images,
            cache=CACHE_DATASET
        )

        xx = 0
        for x, label in enumerate(self.dataset.train_labels):
            #print("LABEL:", label, self.dataset.class_names[label], LCHECK_ORD)
            if self.dataset.class_names[label] == LCHECK_ORD:
                print("FOUND!!!")
                plt.matshow(self.dataset.train_images[x])
                plt.show()
                xx += 1
                if xx > 10:
                    break

    def run(self):
        if CACHE_MODEL:
            self.model = keras.models.load_model(self.model_path)
        else:
            self.model = self.cnn_model()

            if SMALL_SAMPLE_ONLY:
                self.model.save(self.model_path)
            else:
                self.model.save(self.model_path)

    def cnn_model(self):
        x_train, y_train = self.dataset.train_images, \
                           self.dataset.train_labels
        x_train = x_train.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

        # Various resources I used in coming to these parameters:
        # https://github.com/jtyoui/Jtyoui/blob/master/jtyoui/neuralNetwork/kerase/HandWritingRecognition.py (MIT)
        # https://pdfs.semanticscholar.org/4941/aed85462968e9918110b4ba740c56030fd23.pdf
        # "Hands-On Machine Learning with Scikit-Learn and TensorFlow" by Aurelien Geron
        # It might pay to use BatchNormalization and early stopping, too.

        model = self.model = keras.Sequential([
            keras.layers.Convolution2D(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
                                       filters=64,  # Number of outputs
                                       kernel_size=3,
                                       strides=1,
                                       padding='same',
                                       # Maybe linear activation is a better option
                                       # (i.e. commented out?)
                                       #activation='relu'
                                       ),
            keras.layers.Convolution2D(filters=64,
                                       kernel_size=3,
                                       padding='same',
                                       #activation='relu'
                                       ),
            keras.layers.MaxPool2D(pool_size=1,
                                   strides=2,
                                   padding='same'),

            keras.layers.Convolution2D(filters=128,
                                       kernel_size=3,
                                       padding='same',
                                       #activation='relu'
                                       ),
            keras.layers.Convolution2D(filters=128,
                                       kernel_size=3,
                                       padding='same',
                                       #activation='relu'
                                       ),
            keras.layers.MaxPool2D(pool_size=1,
                                   strides=2,
                                   padding='same'),

            keras.layers.Convolution2D(filters=256,
                                       kernel_size=3,
                                       padding='same',
                                       activation='relu'),
            keras.layers.Convolution2D(filters=256,
                                       kernel_size=3,
                                       padding='same',
                                       activation='relu'),
            keras.layers.MaxPool2D(pool_size=1,
                                   strides=2,
                                   padding='same'),

            #keras.layers.Convolution2D(filters=512,
            #                           kernel_size=3,
            #                           padding='same',
            #                           activation='relu'),
            #keras.layers.Convolution2D(filters=512,
            #                           kernel_size=3,
            #                           padding='same',
            #                           activation='relu'),
            #keras.layers.Convolution2D(filters=512,
            #                           kernel_size=3,
            #                           padding='same',
            #                           activation='relu'),
            #keras.layers.MaxPool2D(pool_size=1,
            #                       strides=2,
            #                       padding='same'),

            keras.layers.Flatten(),

            keras.layers.Dense(units=4096,
                               activation='relu'),
            keras.layers.Dropout(0.5),

            keras.layers.Dense(units=2048,
                               activation='relu'),
            keras.layers.Dropout(0.5),

            keras.layers.Dense(units=2048,
                               activation='relu'),
            keras.layers.Dropout(0.5),

            keras.layers.Dense(units=2048,
                               activation='relu'),

            # Perhaps this is not the right place for this?
            # https://stats.stackexchange.com/questions/299292/dropout-makes-performance-worse
            #keras.layers.Dropout(0.5),

            keras.layers.Dense(units=len(self.dataset.class_names),
                               activation=keras.activations.softmax),
        ])
        opt = keras.optimizers.Adam(lr=1e-4)
        model.compile(optimizer=opt,
                      #loss=keras.losses.categorical_crossentropy,
                      loss=keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy', #'mae'
                               ])

        this = self
        class MyCustomCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                this.do_prediction(LCHECK_RASTERED, LCHECK_ORD,
                                   LAugRastered=LCHECK_RASTERED_AUG)

        model.fit(
            x=x_train, y=y_train,
            batch_size=BATCH_SIZE,
            epochs=NUM_EPOCHS,
            callbacks=[MyCustomCallback()]
        )
        return model

    def do_prediction(self, rastered, should_be_ord, LAugRastered=None):
        plt.matshow(rastered)
        plt.show()

        # Convert it to a format the model understands
        rastered = rastered.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
        #rastered = rastered.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
        #print("RASTERED:", rastered)

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
            result = np.zeros(shape=(len(self.dataset.class_names),), dtype='float32')
            for i_rastered in LAugRastered:
                i_rastered = i_rastered.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
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
    #LCHECK = [[[58,18],[58,20],[59,24],[59,26],[62,45],[62,49],[62,54],[62,57],[62,61],[62,66],[62,74],[62,77],[62,81],[62,83],[62,87],[62,90],[62,95],[62,102],[62,108],[62,114],[62,119],[62,123],[62,127],[62,130],[61,134],[61,138],[61,143],[60,147],[60,151],[59,155],[59,159],[59,163],[59,165],[59,166],[59,167],[59,168],[59,169],[59,171],[59,173],[59,176]],[[59,18],[63,28],[65,29],[66,30],[71,30],[107,27],[111,27],[112,27],[113,27],[118,27],[123,26],[131,25],[138,24],[142,23],[145,23],[148,23],[150,23],[154,23],[156,22],[157,22],[158,22],[159,22],[160,22],[164,22],[167,22],[168,22],[168,23],[168,29],[167,36],[166,44],[166,50],[165,54],[165,57],[165,59],[165,61],[164,65],[164,70],[164,75],[164,78],[164,80],[164,83],[164,87],[165,92],[165,99],[166,106],[167,113],[168,119],[168,126],[169,131],[169,136],[169,140],[169,143],[169,146],[169,147],[169,151],[170,156],[170,157],[170,158],[171,160],[172,162],[172,165],[172,166],[172,167],[173,167],[173,170],[175,173],[177,177]],[[63,66],[65,66],[67,66],[70,66],[73,66],[76,67],[83,68],[89,68],[101,65],[109,65],[117,64],[122,64],[127,64],[129,64],[131,64],[132,64],[133,64],[134,64],[137,65],[142,67],[146,68],[151,68],[154,69],[155,70],[156,70],[157,70],[158,70],[159,70],[159,71],[160,71]],[[60,113],[61,113],[63,113],[67,114],[69,114],[72,115],[81,115],[91,115],[100,114],[108,114],[110,114],[112,114],[114,114],[117,114],[121,114],[126,114],[130,114],[135,114],[137,114],[139,114],[142,114],[149,114],[155,114],[162,114],[163,114],[164,114]],[[61,172],[64,172],[65,172],[68,173],[69,173],[71,173],[76,174],[86,175],[94,177],[101,177],[105,177],[106,177],[109,178],[115,178],[122,179],[130,181],[135,181],[139,181],[142,181],[147,181],[154,181],[160,181],[166,180],[167,180],[168,180],[171,180],[171,179],[173,179],[176,179]]]
    #LCHECK = [[(0, 0), (0, 1000)], [(15, 13), (992, 50), (1000, 892)], [(132, 355), (976, 355)], [(46, 630), (1000, 771)], [(0, 979), (992, 979)]]
    #LCHECK = [[[63,20],[63,19],[61,22],[57,26],[52,32],[47,38],[40,43],[22,58],[21,59]],[[24,80],[28,80],[35,80],[43,80],[50,80],[55,80],[58,80],[62,80],[64,80],[70,80],[72,80],[78,80],[83,80],[85,80],[91,80],[94,80],[99,79],[103,79],[106,79],[113,79],[117,79],[123,78],[128,78],[133,77],[139,77],[144,77],[149,77],[156,77],[159,77],[165,76],[174,76],[176,76],[180,76],[185,76],[191,76],[192,76],[195,76],[196,76],[197,76],[198,76],[199,76],[201,76],[202,76],[203,76],[205,76],[210,75],[212,75],[217,73],[219,73],[221,73],[222,73],[223,73],[224,73]],[[49,34],[49,39],[50,50],[50,60],[50,66],[50,68],[50,72],[50,75],[50,78],[50,81],[50,83],[50,85],[50,87],[50,91],[50,95],[50,97],[50,100],[50,103],[50,106],[51,110],[51,112],[51,115],[51,118],[51,121],[53,129],[53,131],[53,134],[53,136],[53,139],[53,142],[53,145],[53,146],[53,148],[53,149],[53,152],[53,155],[53,157],[53,159],[54,162],[54,164],[55,167],[55,169],[55,170],[55,175],[55,176],[55,177],[55,180],[55,181],[55,183],[55,184],[55,185],[55,186],[54,186],[53,186],[50,184],[47,182],[45,181],[42,180]],[[70,119],[70,120],[67,122],[64,123],[62,124],[60,127],[59,128],[56,129],[55,131],[54,132],[53,133],[51,135],[49,136],[48,138],[47,139],[45,141],[44,142],[43,143],[42,145],[41,146],[39,147],[38,148],[36,149],[35,150],[33,151],[32,152],[31,153],[30,154]],[[94,28],[98,31],[107,49],[118,72],[120,80],[126,91],[132,105],[137,115],[141,128],[142,132],[146,140],[149,148],[154,160],[158,171],[160,175],[164,180],[166,184],[170,187],[172,190],[175,195],[177,196],[177,197],[178,198],[178,199],[180,200],[181,200],[183,200]],[[187,126],[186,128],[184,131],[180,134],[178,136],[174,140],[171,142],[168,145],[165,148],[162,149],[157,152],[155,155],[152,156],[151,158],[144,163],[142,165],[141,166],[141,167],[137,170],[134,172],[133,174],[131,175],[130,175],[129,176],[128,176],[127,176],[126,177]],[[155,33],[157,34],[159,35],[162,37],[164,37],[165,39],[167,40],[168,41],[170,42],[172,43],[174,45],[177,48],[179,49],[181,50],[182,52],[183,53],[184,54],[185,55]]]
    LCHECK =    [[(208, 0), (199, 119), (94, 341)], [(0, 461), (781, 520), (915, 520), (999, 479)],
                [(189, 167), (213, 209), (238, 826), (268, 934), (203, 910)], [(303, 514), (94, 766)],
                [(462, 17), (497, 586), (522, 688), (646, 886), (796, 1000)], [(716, 628), (462, 916)],
                [(696, 101), (771, 155), (835, 251)]]
    LCHECK_ORD = ord('我')

    aug = HWStrokesAugmenter(LCHECK)
    LCHECK_RASTERED = aug.raster_strokes(on_val=255, image_size=IMAGE_SIZE, do_augment=False) / 255.0
    plt.matshow(LCHECK_RASTERED)
    plt.show()

    LCHECK_RASTERED_AUG = [
        aug.raster_strokes(on_val=255, image_size=IMAGE_SIZE) / 255.0
        for ___ in range(20)
    ]

    #rastered = rastered.transpose()

    demo = HandwritingModel()
    demo.run()
    demo.do_prediction(LCHECK_RASTERED, LCHECK_ORD,
                       LAugRastered=LCHECK_RASTERED_AUG)

    for x, img in enumerate(demo.dataset.train_images):
        should_be_ord = demo.dataset.class_names[demo.dataset.train_labels[x]]
        print('ENUMERATE x:', x, demo.dataset.train_labels[x],
              chr(should_be_ord))
        demo.do_prediction(img, should_be_ord)
        if x > 30:
            break
