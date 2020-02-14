from configuration import DatasetName, DatasetType, \
    AffectnetConf, IbugConf, W300Conf, InputDataSize, LearningConfig
from tf_record_utility import TFRecordUtility
from clr_callback import CyclicLR
from cnn_model import CNNModel
from custom_Losses import Custom_losses
from Data_custom_generator import Custom_Heatmap_Generator
import tensorflow as tf
import keras

print(tf.__version__)
print(keras.__version__)

tf.logging.set_verbosity(tf.logging.ERROR)
from keras.callbacks import ModelCheckpoint

from keras.optimizers import adam
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.callbacks import CSVLogger
from datetime import datetime
from sklearn.utils import shuffle
import os
from sklearn.model_selection import train_test_split
from numpy import save, load, asarray
import os.path

class Train:
    def __init__(self, use_tf_record, dataset_name, custom_loss, arch, inception_mode, num_output_layers, weight=None):
        cnn = CNNModel()
        c_loss = Custom_losses()

        if dataset_name == DatasetName.ibug:
            self.SUM_OF_ALL_TRAIN_SAMPLES = IbugConf.sum_of_train_samples
        elif dataset_name == DatasetName.affectnet:
            self.SUM_OF_ALL_TRAIN_SAMPLES = AffectnetConf.sum_of_train_samples

        self.BATCH_SIZE = LearningConfig.batch_size
        self.STEPS_PER_VALIDATION_EPOCH = LearningConfig.steps_per_validation_epochs
        self.STEPS_PER_EPOCH = self.SUM_OF_ALL_TRAIN_SAMPLES // self.BATCH_SIZE
        self.EPOCHS = LearningConfig.epochs

        if custom_loss:
            self.loss = c_loss.custom_loss_hm
        else:
            self.loss = keras.losses.mse

        self.arch = arch
        self.inception_mode = inception_mode
        self.weight = weight
        self.num_output_layers = num_output_layers

        if use_tf_record:
            self.train_fit()
        else:
            self.train_fit_gen()


    def train_fit_gen(self):
        """train_fit_gen"""

        '''prepare callbacks'''
        callbacks_list = self._prepare_callback()

        """define optimizers"""
        optimizer = self._get_optimizer()

        '''create train, validation, test data iterator'''
        x_train_filenames, x_val_filenames, y_train, y_val = self._create_generators()
        print('done !!!')

        my_training_batch_generator = Custom_Heatmap_Generator(x_train_filenames, y_train, self.BATCH_SIZE)
        my_validation_batch_generator = Custom_Heatmap_Generator(x_val_filenames, y_val,  self.BATCH_SIZE)

        '''creating model'''
        model = self._get_model(None)

        if self.weight is not None:
            model.load_weights(self.weight)

        '''compiling model'''
        model.compile(loss=self._generate_loss(),
                      optimizer=optimizer,
                      metrics=['mse', 'mae'],
                      loss_weights=self._generate_loss_weights()
                      )

        '''train Model '''
        print('< ========== Start Training ============= >')
        model.fit_generator(generator=my_training_batch_generator,
                            epochs=self.EPOCHS,
                            verbose=1,
                            validation_data=my_validation_batch_generator,
                            steps_per_epoch=self.STEPS_PER_EPOCH,
                            callbacks=callbacks_list,
                            use_multiprocessing=True,
                            workers=16,
                            max_queue_size=32)

    def train_fit(self):
        tf_record_util = TFRecordUtility()

        '''prepare callbacks'''
        callbacks_list = self._prepare_callback()

        ''' define optimizers'''
        optimizer = self._get_optimizer()

        '''create train, validation, test data iterator'''
        train_images, _, _, _, _, _, _, train_heatmap, _ = \
            tf_record_util.create_training_tensor(tfrecord_filename=IbugConf.tf_train_path_heatmap,
                                                  batch_size=self.BATCH_SIZE, reduced=True)
        validation_images, _, _, _, _, _, _, validation_heatmap, _ = \
            tf_record_util.create_training_tensor(tfrecord_filename=IbugConf.tf_evaluation_path_heatmap,
                                                  batch_size=self.BATCH_SIZE, reduced=True)

        '''creating model'''
        model = self._get_model(train_images)

        if self.weight is not None:
            model.load_weights(self.weight)

        '''compiling model'''
        model.compile(loss=self._generate_loss(),
                      optimizer= optimizer,
                      metrics=['mse', 'mae'],
                      target_tensors=self._generate_target_tensors(train_heatmap),
                      loss_weights=self._generate_loss_weights()
                      )

        '''train Model '''
        print('< ========== Start Training ============= >')

        history = model.fit(train_images,
                            train_heatmap,
                            epochs=self.EPOCHS,
                            steps_per_epoch=self.STEPS_PER_EPOCH,
                            validation_data=(validation_images, validation_heatmap),
                            validation_steps=self.STEPS_PER_VALIDATION_EPOCH,
                            verbose=1, callbacks=callbacks_list,
                            use_multiprocessing=True,
                            workers=16,
                            max_queue_size=32
                            )

    def _generate_loss(self):
        loss = []
        for i in range(self.num_output_layers):
            loss.append(self.loss)
        return loss

    def _generate_target_tensors(self, target_tensor):
        tensors = []
        for i in range(self.num_output_layers):
            tensors.append(target_tensor)
        return tensors

    def _generate_loss_weights(self):
        return [1]

    def _get_model(self, train_images):
        cnn = CNNModel()
        if self.arch == 'hg':
            model = cnn.hour_glass_network(num_stacks=self.num_output_layers)
        elif self.arch == '':
            model = cnn.mnv2_hm(tensor=train_images)
        elif self.arch == '':
            model = cnn.mnv2_hm_r_v2(tensor=train_images, inception_mode=self.inception_mode)
        return model

    def _get_optimizer(self):
        return adam(lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-5, amsgrad=False)

    def _prepare_callback(self):
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='min')
        file_path = "weights-{epoch:02d}-{loss:.5f}.h5"
        checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
        csv_logger = CSVLogger('log.csv', append=True, separator=';')

        clr = CyclicLR(
            mode=LearningConfig.CLR_METHOD,
            base_lr=LearningConfig.MIN_LR,
            max_lr=LearningConfig.MAX_LR,
            step_size=LearningConfig.STEP_SIZE * (self.SUM_OF_ALL_TRAIN_SAMPLES // self.BATCH_SIZE))

        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        return [checkpoint, early_stop, csv_logger, clr, tensorboard_callback]

    def _create_file_name_and_label(self):
        images_dir = '/media/ali/extradata/facial_landmark_ds/from_ibug/train_set/train_before_heatmap/'

        subdirs, dirs, files = os.walk(images_dir).__next__()

        filenames = []
        labels = []

        for file in os.listdir(images_dir):
            if file.endswith(".jpg") or file.endswith(".png"):
                file_name = os.path.join(images_dir, file)
                filenames.append(file_name)
                pts_file = file_name[:-3] + "pts"
                points_arr = []
                with open(pts_file) as fp:
                    line = fp.readline()
                    cnt = 1
                    while line:
                        if 3 < cnt < 72:
                            x_y_pnt = line.strip()
                            x = float(x_y_pnt.split(" ")[0])
                            y = float(x_y_pnt.split(" ")[1])
                            points_arr.append(x)
                            points_arr.append(y)
                        line = fp.readline()
                        cnt += 1
                labels.append(points_arr)
        return np.array(filenames), np.array(labels)

    def _create_generators(self):
        if os.path.isfile('x_train_filenames.npy') and \
                os.path.isfile('x_val_filenames.npy') and \
                os.path.isfile('y_train.npy') and \
                os.path.isfile('y_val.npy'):
            x_train_filenames = load('x_train_filenames.npy')
            x_val_filenames = load('x_val_filenames.npy')
            y_train = load('y_train.npy')
            y_val = load('y_val.npy')
        else:
            filenames, labels = self._create_file_name_and_label()

            filenames_shuffled, y_labels_shuffled = shuffle(filenames, labels)

            x_train_filenames, x_val_filenames, y_train, y_val = train_test_split(
                filenames_shuffled, y_labels_shuffled, test_size=0.1, random_state=1)

            save('x_train_filenames.npy', x_train_filenames)
            save('x_val_filenames.npy', x_val_filenames)
            save('y_train.npy', y_train)
            save('y_val.npy', y_val)

        return x_train_filenames, x_val_filenames, y_train, y_val

    def _save_model_with_weight(self, model):
        model_json = model.to_json()

        with open("model_asm_shrink.json", "w") as json_file:
            json_file.write(model_json)

        model.save_weights("model_asm_shrink.h5")
        print("Saved model to disk")
