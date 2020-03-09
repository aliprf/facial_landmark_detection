from configuration import DatasetName, DatasetType, \
    AffectnetConf, IbugConf, W300Conf, InputDataSize, LearningConfig
from tf_record_utility import TFRecordUtility
from clr_callback import CyclicLR
from cnn_model import CNNModel
from custom_Losses import Custom_losses
from data_Heatmap_Generator import DataHeatmapGenerator
from data_pw_generator import DataPWGenerator
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
from keras import losses

class Train:
    def __init__(self, use_tf_record, dataset_name, custom_loss, arch,
                 inception_mode, num_output_layers, point_wise, weight=None):
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
            self.loss = losses.mean_squared_error

        self.arch = arch
        self.inception_mode = inception_mode
        self.weight = weight
        self.num_output_layers = num_output_layers
        self.point_wise = point_wise

        if use_tf_record:
            self.train_fit()
        else:
            if point_wise:
                self.train_fit_gen_point_wise()
            else:
                self.train_fit_gen()

    def train_fit_gen_point_wise(self):
        """train_fit_gen_point_wise"""

        '''prepare callbacks'''
        callbacks_list = self._prepare_callback()

        """define optimizers"""
        optimizer = self._get_optimizer()

        '''create train, validation, test data iterator'''
        x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = self._create_generators()

        my_training_batch_generator = DataPWGenerator(x_train_filenames, y_train_filenames,
                                                      self.BATCH_SIZE, self.num_output_layers)
        my_validation_batch_generator = DataPWGenerator(x_val_filenames, y_val_filenames,
                                                        self.BATCH_SIZE, self.num_output_layers)

        '''creating model'''
        model = self._get_model(None)

        if self.weight is not None:
            model.load_weights(self.weight)

        '''compiling model'''
        model.compile(loss=self._generate_loss(),
                      optimizer=optimizer,
                      metrics=['mse'],
                      loss_weights=self._generate_loss_weights()
                      )

        '''train Model '''
        print('< ========== Start Training ============= >')
        # model.fit(x=[np.ones([30, 224, 224, 3]), np.ones([30, 224, 224, 3])],
        #           y=[np.ones([30, 2]), np.ones([30, 2])],
        #                     epochs=self.EPOCHS,
        #                     verbose=1
        #                     )

        model.fit_generator(generator=[my_training_batch_generator, my_training_batch_generator],
                            epochs=self.EPOCHS,
                            verbose=1,
                            validation_data=[my_validation_batch_generator, my_validation_batch_generator],
                            steps_per_epoch=self.STEPS_PER_EPOCH,
                            callbacks=callbacks_list,
                            use_multiprocessing=True,
                            workers=1,
                            max_queue_size=64
                            )

    def train_fit_gen(self):
        """train_fit_gen"""

        '''prepare callbacks'''
        callbacks_list = self._prepare_callback()

        """define optimizers"""
        optimizer = self._get_optimizer()

        '''create train, validation, test data iterator'''
        x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = self._create_generators()

        my_training_batch_generator = DataHeatmapGenerator(x_train_filenames, y_train_filenames,
                                                           self.BATCH_SIZE, self.num_output_layers)
        my_validation_batch_generator = DataHeatmapGenerator(x_val_filenames, y_val_filenames,
                                                             self.BATCH_SIZE, self.num_output_layers)

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
                            workers=12,
                            max_queue_size=24
                            )

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
                      optimizer=optimizer,
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
        wights = []
        for i in range(self.num_output_layers):
            wights.append(1)
        return wights

    def _get_model(self, train_images):
        cnn = CNNModel()

        if self.arch == 'sh_reg':
            model = cnn.create_shallow_reg()
        if self.arch == 'hg':
            model = cnn.hour_glass_network(num_stacks=self.num_output_layers)
        elif self.arch == 'mn_r':
            model = cnn.mnv2_hm(tensor=train_images)
        elif self.arch == 'mn_main':
            model = cnn.mobileNet_v2_small(tensor=train_images)
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


    def _create_generators(self):
        tf_utils = TFRecordUtility()
        if self.point_wise:
            if os.path.isfile('x_train_filenames_pw.npy') and \
                    os.path.isfile('x_val_filenames_pw.npy') and \
                    os.path.isfile('y_train_filenames_pw.npy') and \
                    os.path.isfile('y_val_filenames_pw.npy'):
                x_train_filenames = load('x_train_filenames_pw.npy')
                x_val_filenames = load('x_val_filenames_pw.npy')
                y_train = load('y_train_filenames_pw.npy')
                y_val = load('y_val_filenames_pw.npy')
            else:
                for i in range(68):
                    filenames, labels = tf_utils.create_fused_images_and_labels_name()

                    filenames_shuffled, y_labels_shuffled = shuffle(filenames, labels)

                    x_train_filenames, x_val_filenames, y_train, y_val = train_test_split(
                        filenames_shuffled, y_labels_shuffled, test_size=0.1, random_state=1)

                    save('npy/' + 'x_train_filenames_pw_.npy', x_train_filenames)
                    save('npy/' + 'x_val_filenames_pw_.npy', x_val_filenames)
                    save('npy/' + 'y_train_filenames_pw_.npy', y_train)
                    save('npy/' + 'y_val_filenames_pw_.npy', y_val)
        else:
            if os.path.isfile('x_train_filenames.npy') and \
                    os.path.isfile('x_val_filenames.npy') and \
                    os.path.isfile('y_train_filenames.npy') and \
                    os.path.isfile('y_val_filenames.npy'):
                x_train_filenames = load('x_train_filenames.npy')
                x_val_filenames = load('x_val_filenames.npy')
                y_train = load('y_train_filenames.npy')
                y_val = load('y_val_filenames.npy')
            else:
                filenames, labels = tf_utils.create_image_and_labels_name(self.point_wise)

                filenames_shuffled, y_labels_shuffled = shuffle(filenames, labels)

                x_train_filenames, x_val_filenames, y_train, y_val = train_test_split(
                    filenames_shuffled, y_labels_shuffled, test_size=0.1, random_state=1)

                save('x_train_filenames.npy', x_train_filenames)
                save('x_val_filenames.npy', x_val_filenames)
                save('y_train_filenames.npy', y_train)
                save('y_val_filenames.npy', y_val)

        return x_train_filenames, x_val_filenames, y_train, y_val

    def _save_model_with_weight(self, model):
        model_json = model.to_json()

        with open("model_asm_shrink.json", "w") as json_file:
            json_file.write(model_json)

        model.save_weights("model_asm_shrink.h5")
        print("Saved model to disk")
