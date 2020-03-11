from configuration import DatasetName, DatasetType, \
    AffectnetConf, IbugConf, W300Conf, InputDataSize, LearningConfig
from hg_Class import HourglassNet
from shallow_reg_network import ShallowRegNetwork

import tensorflow as tf
import keras
from skimage.transform import resize

from keras.regularizers import l2
tf.logging.set_verbosity(tf.logging.ERROR)
from keras.models import Model
from keras.applications import mobilenet_v2, mobilenet, resnet50, densenet
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, \
    BatchNormalization, Activation, GlobalAveragePooling2D, DepthwiseConv2D, \
    Dropout, ReLU, Concatenate, Deconvolution2D, Input

from keras.callbacks import ModelCheckpoint
from keras import backend as K

from keras.optimizers import adam
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.callbacks import CSVLogger
from clr_callback import CyclicLR
from datetime import datetime

import  cv2
import os.path
from keras.utils.vis_utils import plot_model
from scipy.spatial import distance
import scipy.io as sio


class CNNModel:
    def create_shallow_reg(self):
        sh_net = ShallowRegNetwork(inp_shape=32, out_shape=2, num_branches=2, num_filters=32)
        return sh_net.create_model()

    def hour_glass_network(self, num_classes=68, num_stacks=4, num_channels=512, in_shape=(224, 224), out_shape=(56, 56)):
        hg_net = HourglassNet(num_classes=num_classes, num_stacks=num_stacks,
                              num_channels=num_channels,
                              inres=in_shape,
                              outres=out_shape)
        model = hg_net.build_model()
        return model

    def shrink_v1_mobileNet_v2_multi_task(self, tensor):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)

        mobilenet_model.layers.pop()

        inp = mobilenet_model.input

        # res 1
        block_1_project_BN = mobilenet_model.get_layer('block_1_project_BN').output  # 56, 56, 24
        dasm_1_conv_2d = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                use_bias=False,
                                name='dasm_1_conv_2d', kernel_initializer='normal')(block_1_project_BN)
        dasm_1_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                       name='dasm_1_BN')(dasm_1_conv_2d)  # 28, 28, 32
        dasm_1_relu = ReLU(6., name='dasm_1_relu')(dasm_1_BN)

        block_3_project_BN = mobilenet_model.get_layer('block_3_project_BN').output  # 28, 28, 32
        dasm_combined_1_and_3_layer = keras.layers.add([dasm_1_relu, block_3_project_BN])  # 28, 28, 32

        # rev_res 1
        x = Conv2D(6 * 32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='dasm_combined_1_and_3_layer_expand')(dasm_combined_1_and_3_layer)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='dasm_combined_1_and_3_layer_expand_BN')(x)
        x = ReLU(6., name='dasm_combined_1_and_3_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='dasm_combined_1_and_3_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='dasm_combined_1_and_3_layer_depthwise_BN')(x)
        x = ReLU(6., name='dasm_combined_1_and_3_layer_depthwise_BN_relu')(x)
        x = Conv2D(32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='dasm_combined_1_and_3_layer_project')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='dasm_combined_1_and_3_layer_project_project_BN')(x)  # 28, 28, 32
        x = ReLU(6., name='dasm_combined_1_and_3_layer_project_project_BN_relu')(x)
        y = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                   use_bias=False,
                   name='dasm_1_and_3_reduce_conv_2d', kernel_initializer='normal')(x)
        dasm_1_and_3_reduce_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='dasm_1_and_3_reduce_BN')(y)

        # res 3
        block_9_project_BN = mobilenet_model.get_layer('block_9_project_BN').output  # 14, 14, 96
        dasm_combined_1_3_and_9_layer = keras.layers.add([block_9_project_BN,
                                                          dasm_1_and_3_reduce_BN])  # 14, 14, 96

        x = Conv2D(128, kernel_size=1, use_bias=False, name='Conv_1')(dasm_combined_1_3_and_9_layer)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
        x = ReLU(6., name='out_relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(LearningConfig.landmark_len, name='dense_layer_out_2', activation='relu'
                  , kernel_initializer='he_uniform')(x)

        # multitask output layers
        # Logits_pose = Dense(InputDataSize.pose_len, name='out_pose')(x)
        Logits = Dense(LearningConfig.landmark_len, name='out_main')(x)
        Logits_face = Dense(InputDataSize.landmark_face_len, name='out_face')(x)
        Logits_nose = Dense(InputDataSize.landmark_nose_len, name='out_nose')(x)
        Logits_eyes = Dense(InputDataSize.landmark_eys_len, name='out_eyes')(x)
        Logits_mouth = Dense(InputDataSize.landmark_mouth_len, name='out_mouth')(x)

        revised_model = Model(inp, [Logits, Logits_nose, Logits_eyes, Logits_face, Logits_mouth])
        # revised_model = Model(inp, [Logits, Logits_nose, Logits_eyes, Logits_face, Logits_mouth, Logits_pose])
        return revised_model

    def test_model(self):
        input_1 = Input(shape=(224, 224, 3))

        conv1 = Conv2D(16, (3, 3), activation='relu', padding="SAME")(input_1)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)
        conv2 = Conv2D(32, (3, 3), activation='relu', padding="SAME")(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)
        flat_1 = Flatten()(pool2)

        # feature extraction from RGB image
        inputs_2 = Input(shape=(224, 224, 3))

        conv1_2 = Conv2D(16, (3, 3), activation='relu', padding="SAME")(inputs_2)
        pool1_2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1_2)
        conv2_2 = Conv2D(32, (3, 3), activation='relu', padding="SAME")(pool1_2)
        pool2_2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2_2)
        flat_2 = Flatten()(pool2_2)

        # concatenate both feature layers and define output layer after some dense layers
        concat = Concatenate()([flat_1, flat_2])
        dense1 = Dense(60, activation='relu')(concat)
        dense2 = Dense(60, activation='relu')(dense1)
        dense3 = Dense(60, activation='relu')(dense2)
        output_1 = Dense(2)(dense3)
        output_2 = Dense(2)(dense3)

        # create model with two inputs
        model = Model([input_1, inputs_2], [output_1, output_2])
        model_json = model.to_json()

        with open("test_model.json", "w") as json_file:
            json_file.write(model_json)

        model.summary()
        return model


    def mobileNet_v2_small(self, tensor):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=[56, 56, 68],
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)

        mobilenet_model.layers.pop()

        x = mobilenet_model.get_layer('global_average_pooling2d_1').output  # 1280
        x = Dense(LearningConfig.landmark_len, name='dense_layer_out_2', activation='relu',
                  kernel_initializer='he_uniform')(x)
        out = Dense(LearningConfig.landmark_len, name='out')(x)

        inp = mobilenet_model.input

        revised_model = Model(inp, out)

        revised_model.summary()
        # plot_model(revised_model, to_file='mobileNet_v2_main.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("0mobileNet_v2_small.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def shrink_v2_mobileNet_v2_multi_task(self, tensor, pose):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)

        mobilenet_model.layers.pop()

        inp = mobilenet_model.input

        '''------------------------------------'''
        '''residual network_branch_1'''
        '''     block_1  '''
        r1_block_1_project_BN = mobilenet_model.get_layer('block_1_project_BN').output  # 56, 56, 24
        r1_dasm_1_conv_2d = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                   use_bias=False,
                                   name='r1_dasm_1_conv_2d', kernel_initializer='normal')(r1_block_1_project_BN)
        r1_dasm_1_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                          name='r1_dasm_1_BN')(r1_dasm_1_conv_2d)  # 28, 28, 32
        # r1_dasm_1_relu = ReLU(6., name='r1_dasm_1_relu')(r1_dasm_1_BN)

        block_3_project_BN = mobilenet_model.get_layer('block_3_project_BN').output  # 28, 28, 32
        r1_dasm_combined_1_and_3_layer = keras.layers.add([r1_dasm_1_BN, block_3_project_BN])  # 28, 28, 32

        '''     block_2 '''
        x = Conv2D(6 * 32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_and_3_layer_expand')(r1_dasm_combined_1_and_3_layer)  # 28, 28, 6*32=192
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r1_dasm_combined_1_and_3_layer_expand_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r1_dasm_combined_1_and_3_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_and_3_layer_depthwise_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_depthwise_BN_relu')(x)

        x = Conv2D(32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_and_3_layer_project')(x)  # 28, 28, 6*32=192
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_and_3_layer_project_project_BN')(x)  # 28, 28, 32

        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_project_project_BN_relu')(x)

        y = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                   use_bias=False,
                   name='r1_dasm_1_and_3_reduce_conv_2d', kernel_initializer='normal')(x)  # 14, 14, 64
        r1_dasm_1_and_3_reduce_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                                       name='r1_dasm_1_and_3_reduce_BN')(y)
        # r1_dasm_1_and_3_reduce_BN_relu = ReLU(6., name='r1_dasm_1_and_3_reduce_BN_relu')(y)  # 14, 14, 64

        block_6_project_BN = mobilenet_model.get_layer('block_6_project_BN').output  # 14, 14, 64
        dasm_combined_1_3_and_6_layer = keras.layers.add([block_6_project_BN,
                                                          r1_dasm_1_and_3_reduce_BN])  # 14, 14, 64
        '''     block_3 '''
        x = Conv2D(6 * 64, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_3_and_6_layer_expand')(dasm_combined_1_3_and_6_layer)  # 14, 14, 6*64=384
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r1_dasm_combined_1_3_and_6_layer_expand_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_3_and_6_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r1_dasm_combined_1_3_and_6_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_3_and_6_layer_depthwise_BN')(x)  # 14, 14, 64
        x = ReLU(6., name='r1_dasm_combined_1_3_and_6_layer_depthwise_relu')(x)
        x = Conv2D(64, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_3_and_6_layer_conv2d')(x)  # 14, 14, 64
        r1_dasm_combined_1_3_and_6_layer_expand_BN_expand_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                                                                  name='r1_dasm_combined_1_3_and_6_layer_conv2d_BN')(
            x)  # 14, 14, 64
        '''------------------------------------'''
        '''residual network_branch_2'''
        # res_line_2
        block_1_project_BN = mobilenet_model.get_layer('block_1_project_BN').output  # 56, 56, 24
        dasm_1_conv_2d = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                use_bias=False,
                                name='r2_dasm_1_conv_2d', kernel_initializer='normal')(block_1_project_BN)
        dasm_1_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                       name='r2_dasm_1_BN')(dasm_1_conv_2d)  # 28, 28, 32
        # dasm_1_relu = ReLU(6., name='r2_dasm_1_relu')(dasm_1_BN)

        block_3_project_BN = mobilenet_model.get_layer('block_3_project_BN').output  # 28, 28, 32
        dasm_combined_1_and_3_layer = keras.layers.add([dasm_1_BN, block_3_project_BN])  # 28, 28, 32

        # rev_res 1
        x = Conv2D(6 * 32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r2_dasm_combined_1_and_3_layer_expand')(dasm_combined_1_and_3_layer)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r2_dasm_combined_1_and_3_layer_expand_BN')(x)
        x = ReLU(6., name='r2_dasm_combined_1_and_3_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r2_dasm_combined_1_and_3_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r2_dasm_combined_1_and_3_layer_depthwise_BN')(x)
        x = ReLU(6., name='r2_dasm_combined_1_and_3_layer_depthwise_BN_relu')(x)
        x = Conv2D(32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r2_dasm_combined_1_and_3_layer_project')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='dasm_combined_1_and_3_layer_project_project_BN')(x)  # 28, 28, 32
        x = ReLU(6., name='r2_dasm_combined_1_and_3_layer_project_project_BN_relu')(x)
        y = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                   use_bias=False,
                   name='r2_dasm_1_and_3_reduce_conv_2d', kernel_initializer='normal')(x)
        dasm_1_and_3_reduce_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                                    name='r2_dasm_1_and_3_reduce_BN')(y)

        # res 3
        block_9_project_BN = mobilenet_model.get_layer('block_9_project_BN').output  # 14, 14, 64
        dasm_1_3_and_dasm_1_3_6_and_9_layer = keras.layers.add([block_9_project_BN,
                                                                dasm_1_and_3_reduce_BN,
                                                                r1_dasm_combined_1_3_and_6_layer_expand_BN_expand_BN])  # 14, 14, 64

        '''reducing to 7'''
        # Conv_0 = Conv2D(128, kernel_size=1, padding='same', use_bias=False, activation=None,
        #            name='Conv_0')(dasm_1_3_and_dasm_1_3_6_and_9_layer)  # 14, 14, 64
        Conv_0 = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same',
                        use_bias=False,
                        name='Conv_0', kernel_initializer='normal')(dasm_1_3_and_dasm_1_3_6_and_9_layer)  # 7, 7, 128
        Conv_0_BN = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_0_BN')(Conv_0)  # 14, 14, 64
        Conv_0_ReLU = ReLU(6., name='Conv_0_ReLU')(Conv_0_BN)
        # ..
        x = Conv2D(256, kernel_size=1, use_bias=False, name='Conv_1')(Conv_0_ReLU)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
        x = ReLU(6., name='out_relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, name='dense_layer_out_2', activation='relu'
                  , kernel_initializer='he_uniform')(x)

        # multitask output layers
        Logits = Dense(LearningConfig.landmark_len, name='out_main')(x)
        Logits_face = Dense(InputDataSize.landmark_face_len, name='out_face')(x)
        Logits_nose = Dense(InputDataSize.landmark_nose_len, name='out_nose')(x)
        Logits_eyes = Dense(InputDataSize.landmark_eys_len, name='out_eyes')(x)
        Logits_mouth = Dense(InputDataSize.landmark_mouth_len, name='out_mouth')(x)

        out_pose = Dense(InputDataSize.pose_len, name='out_pose')(x)

        if pose:
            revised_model = Model(inp, [Logits, Logits_nose, Logits_eyes, Logits_face, Logits_mouth, out_pose])
        else:
            revised_model = Model(inp, [Logits, Logits_nose, Logits_eyes, Logits_face, Logits_mouth])

        return revised_model

    def shrink_v3_mobileNet_v2_multi_task(self, tensor, pose):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)

        mobilenet_model.layers.pop()

        inp = mobilenet_model.input

        '''------------------------------------'''
        '''residual network_branch_1'''
        '''     block_1  '''
        r1_block_1_project_BN = mobilenet_model.get_layer('block_1_project_BN').output  # 56, 56, 24
        r1_dasm_1_conv_2d = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                use_bias=False,
                                name='r1_dasm_1_conv_2d', kernel_initializer='normal')(r1_block_1_project_BN)
        r1_dasm_1_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                       name='r1_dasm_1_BN')(r1_dasm_1_conv_2d)  # 28, 28, 32
        # r1_dasm_1_relu = ReLU(6., name='r1_dasm_1_relu')(r1_dasm_1_BN)

        block_3_project_BN = mobilenet_model.get_layer('block_3_project_BN').output  # 28, 28, 32
        r1_dasm_combined_1_and_3_layer = keras.layers.add([r1_dasm_1_BN, block_3_project_BN])  # 28, 28, 32

        '''     block_2 '''
        x = Conv2D(6 * 32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_and_3_layer_expand')(r1_dasm_combined_1_and_3_layer)  # 28, 28, 6*32=192
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r1_dasm_combined_1_and_3_layer_expand_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r1_dasm_combined_1_and_3_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_and_3_layer_depthwise_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_depthwise_BN_relu')(x)

        x = Conv2D(32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_and_3_layer_project')(x)  # 28, 28, 6*32=192
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_and_3_layer_project_project_BN')(x)  # 28, 28, 32

        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_project_project_BN_relu')(x)

        y = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                   use_bias=False,
                   name='r1_dasm_1_and_3_reduce_conv_2d', kernel_initializer='normal')(x)  # 14, 14, 64
        r1_dasm_1_and_3_reduce_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_1_and_3_reduce_BN')(y)
        # r1_dasm_1_and_3_reduce_BN_relu = ReLU(6., name='r1_dasm_1_and_3_reduce_BN_relu')(y)  # 14, 14, 64

        block_6_project_BN = mobilenet_model.get_layer('block_6_project_BN').output  # 14, 14, 64
        dasm_combined_1_3_and_6_layer = keras.layers.add([block_6_project_BN,
                                                          r1_dasm_1_and_3_reduce_BN])  # 14, 14, 64
        '''     block_3 '''
        x = Conv2D(6 * 64, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_3_and_6_layer_expand')(dasm_combined_1_3_and_6_layer)  # 14, 14, 6*64=384
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r1_dasm_combined_1_3_and_6_layer_expand_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_3_and_6_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r1_dasm_combined_1_3_and_6_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_3_and_6_layer_depthwise_BN')(x)  # 14, 14, 64
        x = ReLU(6., name='r1_dasm_combined_1_3_and_6_layer_depthwise_relu')(x)
        x = Conv2D(64, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_3_and_6_layer_conv2d')(x)  # 14, 14, 64
        r1_dasm_combined_1_3_and_6_layer_expand_BN_expand_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_3_and_6_layer_conv2d_BN')(x)  # 14, 14, 64
        '''------------------------------------'''
        '''residual network_branch_2'''
        # res_line_2
        block_1_project_BN = mobilenet_model.get_layer('block_1_project_BN').output  # 56, 56, 24
        dasm_1_conv_2d = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                use_bias=False,
                                name='r2_dasm_1_conv_2d', kernel_initializer='normal')(block_1_project_BN)
        dasm_1_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                       name='r2_dasm_1_BN')(dasm_1_conv_2d)  # 28, 28, 32
        # dasm_1_relu = ReLU(6., name='r2_dasm_1_relu')(dasm_1_BN)

        block_3_project_BN = mobilenet_model.get_layer('block_3_project_BN').output  # 28, 28, 32
        dasm_combined_1_and_3_layer = keras.layers.add([dasm_1_BN, block_3_project_BN])  # 28, 28, 32

        # rev_res 1
        x = Conv2D(6 * 32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r2_dasm_combined_1_and_3_layer_expand')(dasm_combined_1_and_3_layer)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r2_dasm_combined_1_and_3_layer_expand_BN')(x)
        x = ReLU(6., name='r2_dasm_combined_1_and_3_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r2_dasm_combined_1_and_3_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r2_dasm_combined_1_and_3_layer_depthwise_BN')(x)
        x = ReLU(6., name='r2_dasm_combined_1_and_3_layer_depthwise_BN_relu')(x)
        x = Conv2D(32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r2_dasm_combined_1_and_3_layer_project')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='dasm_combined_1_and_3_layer_project_project_BN')(x)  # 28, 28, 32
        x = ReLU(6., name='r2_dasm_combined_1_and_3_layer_project_project_BN_relu')(x)
        y = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                   use_bias=False,
                   name='r2_dasm_1_and_3_reduce_conv_2d', kernel_initializer='normal')(x)
        dasm_1_and_3_reduce_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r2_dasm_1_and_3_reduce_BN')(y)

        # res 3
        block_9_project_BN = mobilenet_model.get_layer('block_9_project_BN').output  # 14, 14, 64
        dasm_1_3_and_dasm_1_3_6_and_9_layer = keras.layers.add([block_9_project_BN,
                                                                dasm_1_and_3_reduce_BN,
                                                                r1_dasm_combined_1_3_and_6_layer_expand_BN_expand_BN])  # 14, 14, 64

        '''reducing to 7'''
        Conv_0 = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same',
                   use_bias=False,
                   name='Conv_0', kernel_initializer='normal')(dasm_1_3_and_dasm_1_3_6_and_9_layer)  # 7, 7, 128

        # Conv_0 = Conv2D(128, kernel_size=1, padding='same', use_bias=False, activation=None,
        #            name='Conv_0')(dasm_1_3_and_dasm_1_3_6_and_9_layer)  # 14, 14, 64
        Conv_0_BN = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_0_BN')(Conv_0)  # 14, 14, 64
        Conv_0_ReLU = ReLU(6., name='Conv_0_ReLU')(Conv_0_BN)
        #..
        x = Conv2D(256, kernel_size=1, use_bias=False, name='Conv_1')(Conv_0_ReLU)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
        x = ReLU(6., name='out_relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, name='dense_layer_out_2', activation='relu'
                  , kernel_initializer='he_uniform')(x)

        # multitask output layers
        Logits = Dense(LearningConfig.landmark_len, name='out_main')(x)
        Logits_face = Dense(InputDataSize.landmark_face_len, name='out_face')(x)
        Logits_nose = Dense(InputDataSize.landmark_nose_len, name='out_nose')(x)
        Logits_eyes = Dense(InputDataSize.landmark_eys_len, name='out_eyes')(x)
        Logits_mouth = Dense(InputDataSize.landmark_mouth_len, name='out_mouth')(x)

        out_pose = Dense(InputDataSize.pose_len, name='out_pose')(x)

        if pose:
            revised_model = Model(inp, [Logits, Logits_nose, Logits_eyes, Logits_face, Logits_mouth,out_pose])
        else:
            revised_model = Model(inp, [Logits, Logits_nose, Logits_eyes, Logits_face, Logits_mouth])

        return revised_model

    def shrink_v4_mobileNet_v2_single_task(self, tensor, pose, test):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)

        mobilenet_model.layers.pop()

        inp = mobilenet_model.input

        '''------------------------------------'''
        '''residual network_branch_1'''
        '''     block_1  '''
        r1_block_1_project_BN = mobilenet_model.get_layer('block_1_project_BN').output  # 56, 56, 24
        r1_dasm_1_conv_2d = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                   use_bias=False,
                                   name='r1_dasm_1_conv_2d', kernel_initializer='normal')(r1_block_1_project_BN)
        r1_dasm_1_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                          name='r1_dasm_1_BN')(r1_dasm_1_conv_2d)  # 28, 28, 32
        # r1_dasm_1_relu = ReLU(6., name='r1_dasm_1_relu')(r1_dasm_1_BN)

        block_3_project_BN = mobilenet_model.get_layer('block_3_project_BN').output  # 28, 28, 32
        r1_dasm_combined_1_and_3_layer = keras.layers.add([r1_dasm_1_BN, block_3_project_BN])  # 28, 28, 32

        '''     block_2 '''
        x = Conv2D(6 * 32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_and_3_layer_expand')(r1_dasm_combined_1_and_3_layer)  # 28, 28, 6*32=192
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r1_dasm_combined_1_and_3_layer_expand_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r1_dasm_combined_1_and_3_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_and_3_layer_depthwise_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_depthwise_BN_relu')(x)

        x = Conv2D(32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_and_3_layer_project')(x)  # 28, 28, 6*32=192
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_and_3_layer_project_project_BN')(x)  # 28, 28, 32

        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_project_project_BN_relu')(x)

        y = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                   use_bias=False,
                   name='r1_dasm_1_and_3_reduce_conv_2d', kernel_initializer='normal')(x)  # 14, 14, 64
        r1_dasm_1_and_3_reduce_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                                       name='r1_dasm_1_and_3_reduce_BN')(y)
        # r1_dasm_1_and_3_reduce_BN_relu = ReLU(6., name='r1_dasm_1_and_3_reduce_BN_relu')(y)  # 14, 14, 64

        block_6_project_BN = mobilenet_model.get_layer('block_6_project_BN').output  # 14, 14, 64
        dasm_combined_1_3_and_6_layer = keras.layers.add([block_6_project_BN,
                                                          r1_dasm_1_and_3_reduce_BN])  # 14, 14, 64
        '''     block_3 '''
        x = Conv2D(6 * 64, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_3_and_6_layer_expand')(dasm_combined_1_3_and_6_layer)  # 14, 14, 6*64=384
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r1_dasm_combined_1_3_and_6_layer_expand_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_3_and_6_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r1_dasm_combined_1_3_and_6_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_3_and_6_layer_depthwise_BN')(x)  # 14, 14, 64
        x = ReLU(6., name='r1_dasm_combined_1_3_and_6_layer_depthwise_relu')(x)
        x = Conv2D(64, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_3_and_6_layer_conv2d')(x)  # 14, 14, 64
        r1_dasm_combined_1_3_and_6_layer_expand_BN_expand_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                                                                  name='r1_dasm_combined_1_3_and_6_layer_conv2d_BN')(
            x)  # 14, 14, 64
        '''------------------------------------'''
        '''residual network_branch_2'''
        # res_line_2
        block_1_project_BN = mobilenet_model.get_layer('block_1_project_BN').output  # 56, 56, 24
        dasm_1_conv_2d = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                use_bias=False,
                                name='r2_dasm_1_conv_2d', kernel_initializer='normal')(block_1_project_BN)
        dasm_1_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                       name='r2_dasm_1_BN')(dasm_1_conv_2d)  # 28, 28, 32
        # dasm_1_relu = ReLU(6., name='r2_dasm_1_relu')(dasm_1_BN)

        block_3_project_BN = mobilenet_model.get_layer('block_3_project_BN').output  # 28, 28, 32
        dasm_combined_1_and_3_layer = keras.layers.add([dasm_1_BN, block_3_project_BN])  # 28, 28, 32

        # rev_res 1
        x = Conv2D(6 * 32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r2_dasm_combined_1_and_3_layer_expand')(dasm_combined_1_and_3_layer)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r2_dasm_combined_1_and_3_layer_expand_BN')(x)
        x = ReLU(6., name='r2_dasm_combined_1_and_3_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r2_dasm_combined_1_and_3_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r2_dasm_combined_1_and_3_layer_depthwise_BN')(x)
        x = ReLU(6., name='r2_dasm_combined_1_and_3_layer_depthwise_BN_relu')(x)
        x = Conv2D(32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r2_dasm_combined_1_and_3_layer_project')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='dasm_combined_1_and_3_layer_project_project_BN')(x)  # 28, 28, 32
        x = ReLU(6., name='r2_dasm_combined_1_and_3_layer_project_project_BN_relu')(x)
        y = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                   use_bias=False,
                   name='r2_dasm_1_and_3_reduce_conv_2d', kernel_initializer='normal')(x)
        dasm_1_and_3_reduce_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                                    name='r2_dasm_1_and_3_reduce_BN')(y)

        # res 3
        block_9_project_BN = mobilenet_model.get_layer('block_9_project_BN').output  # 14, 14, 64
        dasm_1_3_and_dasm_1_3_6_and_9_layer = keras.layers.add([block_9_project_BN,
                                                                dasm_1_and_3_reduce_BN,
                                                                r1_dasm_combined_1_3_and_6_layer_expand_BN_expand_BN])  # 14, 14, 64

        '''reducing to 7'''
        Conv_0 = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same',
                        use_bias=False,
                        name='Conv_0', kernel_initializer='normal')(dasm_1_3_and_dasm_1_3_6_and_9_layer)  # 7, 7, 128
        Conv_0_BN = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_0_BN')(Conv_0)  # 14, 14, 64
        Conv_0_ReLU = ReLU(6., name='Conv_0_ReLU')(Conv_0_BN)

        x = Conv2D(256, kernel_size=1, use_bias=False, name='Conv_1')(Conv_0_ReLU)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
        x = ReLU(6., name='out_relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, name='dense_layer_out_2', activation='relu', kernel_initializer='he_uniform')(x)

        # custom activation:
        if test:  # use batch size 1
            Logits_asm = Dense(LearningConfig.landmark_len, activation=self.custom_activation_test, name='asm_out')(x)
        else:  # use batch size real
            Logits_asm = Dense(LearningConfig.landmark_len, activation=self.custom_activation, name='asm_out')(x)

        # Logits_asm = Dense(LearningConfig.landmark_len, name='out_main')(x)

        out_pose = Dense(InputDataSize.pose_len, name='out_pose')(x)

        if pose:
            revised_model = Model(inp, [Logits_asm, out_pose])
        else:
            revised_model = Model(inp, [Logits_asm])
        return revised_model

    def inception_v2_single_task(self, tensor, pose, multitask):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)
        mobilenet_model.layers.pop()

        inp = mobilenet_model.input

        '''------------------------------------'''
        '''residual network_branch_1'''
        '''     block_1  '''
        r1_block_1_project_BN = mobilenet_model.get_layer('block_1_project_BN').output  # 56, 56, 24
        r1_dasm_1_conv_2d = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                   use_bias=False,
                                   name='r1_dasm_1_conv_2d', kernel_initializer='normal')(r1_block_1_project_BN)
        r1_dasm_1_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                          name='r1_dasm_1_BN')(r1_dasm_1_conv_2d)  # 28, 28, 32
        # r1_dasm_1_relu = ReLU(6., name='r1_dasm_1_relu')(r1_dasm_1_BN)

        block_3_project_BN = mobilenet_model.get_layer('block_3_project_BN').output  # 28, 28, 32
        r1_dasm_combined_1_and_3_layer = keras.layers.add([r1_dasm_1_BN, block_3_project_BN])  # 28, 28, 32
        # r1_dasm_combined_1_and_3_layer = Concatenate()([r1_dasm_1_BN, block_3_project_BN])  # 28, 28, 32

        '''{ first output '''
        out_1 = ReLU(6., name='out_1_relu')(r1_dasm_combined_1_and_3_layer)
        out_1 = GlobalAveragePooling2D(name='out_1_GlobalAveragePooling2D')(out_1)
        out_1 = Dense(LearningConfig.landmark_len + InputDataSize.pose_len, activation='relu', kernel_initializer='he_uniform')(out_1)
        # out_1 = Dropout(rate=0.3)(out_1)
        out_1_face = Dense(LearningConfig.landmark_len, name='out_1_face', kernel_initializer='he_uniform')(out_1)
        out_1_pose = Dense(InputDataSize.pose_len, name='out_1_pose', kernel_initializer='he_uniform')(out_1)

        '''first output }'''


        '''     block_2 '''
        x = Conv2D(6 * 32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_and_3_layer_expand')(r1_dasm_combined_1_and_3_layer)  # 28, 28, 6*32=192
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r1_dasm_combined_1_and_3_layer_expand_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r1_dasm_combined_1_and_3_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_and_3_layer_depthwise_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_depthwise_BN_relu')(x)

        x = Conv2D(32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_and_3_layer_project')(x)  # 28, 28, 6*32=192
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_and_3_layer_project_project_BN')(x)  # 28, 28, 32

        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_project_project_BN_relu')(x)

        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                   use_bias=False,
                   name='r1_dasm_1_and_3_reduce_conv_2d', kernel_initializer='normal')(x)  # 14, 14, 64

        r1_dasm_1_and_3_reduce_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                                       name='r1_dasm_1_and_3_reduce_BN')(x)
        # r1_dasm_1_and_3_reduce_BN_relu = ReLU(6., name='r1_dasm_1_and_3_reduce_BN_relu')(y)  # 14, 14, 64

        block_6_project_BN = mobilenet_model.get_layer('block_6_project_BN').output  # 14, 14, 64
        dasm_combined_1_3_and_6_layer = keras.layers.add([block_6_project_BN, r1_dasm_1_and_3_reduce_BN])  # 14, 14, 64
        # dasm_combined_1_3_and_6_layer = Concatenate()([block_6_project_BN, r1_dasm_1_and_3_reduce_BN])  # 14, 14, 64

        '''{ second output '''
        out_2 = ReLU(6., name='out_2_relu')(dasm_combined_1_3_and_6_layer)
        out_2 = GlobalAveragePooling2D(name='out_2_GlobalAveragePooling2D')(out_2)
        out_2 = Dense(LearningConfig.landmark_len + InputDataSize.pose_len, activation='relu', kernel_initializer='he_uniform')(out_2)
        # out_2 = Dropout(rate=0.3)(out_2)
        out_2_face = Dense(LearningConfig.landmark_len, name='out_2_face', kernel_initializer='he_uniform')(out_2)
        out_2_pose = Dense(InputDataSize.pose_len, name='out_2_pose', kernel_initializer='he_uniform')(out_2)

        '''second output }'''

        '''     block_3 '''
        x = Conv2D(6 * 64, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_3_and_6_layer_expand')(dasm_combined_1_3_and_6_layer)  # 14, 14, 6*64=384
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r1_dasm_combined_1_3_and_6_layer_expand_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_3_and_6_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r1_dasm_combined_1_3_and_6_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_3_and_6_layer_depthwise_BN')(x)  # 14, 14, 64
        x = ReLU(6., name='r1_dasm_combined_1_3_and_6_layer_depthwise_relu')(x)
        x = Conv2D(64, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_3_and_6_layer_conv2d')(x)  # 14, 14, 64
        r1_dasm_combined_1_3_and_6_layer_expand_BN_expand_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                                                                  name='r1_dasm_combined_1_3_and_6_layer_conv2d_BN')(
            x)  # 14, 14, 64
        '''------------------------------------'''

        # res 3
        block_9_add = mobilenet_model.get_layer('block_9_add').output  # 14, 14, 64
        # dasm_1_3_and_dasm_1_3_6_and_9_layer = Concatenate()([block_9_add,r1_dasm_combined_1_3_and_6_layer_expand_BN_expand_BN])  # 14, 14, 64
        dasm_1_3_and_dasm_1_3_6_and_9_layer = keras.layers.add([block_9_add,r1_dasm_combined_1_3_and_6_layer_expand_BN_expand_BN])  # 14, 14, 64

        '''{ third output '''
        out_3 = ReLU(6., name='out_3_relu')(dasm_1_3_and_dasm_1_3_6_and_9_layer)
        out_3 = GlobalAveragePooling2D(name='out_3_GlobalAveragePooling2D')(out_3)
        out_3 = Dense(LearningConfig.landmark_len + InputDataSize.pose_len, activation='relu', kernel_initializer='he_uniform')(out_3)
        out_3 = Dropout(rate=0.3)(out_3)
        out_3_face = Dense(LearningConfig.landmark_len, name='out_3_face', kernel_initializer='he_uniform')(out_3)
        out_3_pose = Dense(InputDataSize.pose_len, name='out_3_pose', kernel_initializer='he_uniform')(out_3)
        '''third output }'''

        '''reducing to 7'''
        Conv_0 = Conv2D(filters=96, kernel_size=(3, 3), strides=(2, 2), padding='same',
                        use_bias=False,
                        name='Conv_0', kernel_initializer='normal')(dasm_1_3_and_dasm_1_3_6_and_9_layer)  # 7, 7, 128
        Conv_0_BN = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_0_BN')(Conv_0)  # 7, 7, 128
        Conv_0_ReLU = ReLU(6., name='Conv_0_ReLU')(Conv_0_BN)

        # --last block of mobilenet
        x = Conv2D(576, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='Conv2D_10_expand')(dasm_1_3_and_dasm_1_3_6_and_9_layer)  # 7, 7, 6*96= 576

        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='10_bn_1')(x)
        x = ReLU(6., name='10_rl_1')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='10_dw')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,name='10_bn_2')(x)
        x = ReLU(6., name='10_rl_2')(x)

        x = Conv2D(192, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='Conv2D_10_project')(x)  # 7, 7,576/3=192
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='10_bn_3')(x)  #
        x = Conv2D(768, kernel_size=1, padding='valid', use_bias=False, activation=None,
                   name='Conv2D_10_conv1')(x)  # 7, 7,192*4=768

        x = BatchNormalization(epsilon=1e-3, momentum=0.999,name='10_bn_4')(x)

        x = ReLU(6., name='10_rl_3')(x)
        x = GlobalAveragePooling2D()(x)

        x = Dense(LearningConfig.landmark_len+InputDataSize.pose_len,
                  name='dense_layer_out_3', activation='relu',
                  kernel_initializer='he_uniform')(x)

        x = Dropout(rate=0.3)(x)

        out_facial = Dense(LearningConfig.landmark_len, name='out')(x)  # data become between -1, 1
        out_pose = Dense(InputDataSize.pose_len, name='out_pose')(x)

        if multitask:
            # multitask output layers
            out_face = Dense(InputDataSize.landmark_face_len, name='out_face')(x)
            out_nose = Dense(InputDataSize.landmark_nose_len, name='out_nose')(x)
            out_eyes = Dense(InputDataSize.landmark_eys_len, name='out_eyes')(x)
            out_mouth = Dense(InputDataSize.landmark_mouth_len, name='out_mouth')(x)

            if pose:
                revised_model = Model(inp, [out_facial, out_pose, out_1, out_2, out_3, out_nose, out_eyes, out_face,
                                            out_mouth])
            else:
                revised_model = Model(inp, [out_facial, out_1, out_2, out_3, out_nose, out_eyes, out_face, out_mouth])
        else:
            if pose:
                revised_model = Model(inp, [out_facial, out_pose,
                                            out_1_face,# out_1_pose,
                                            out_2_face,# out_2_pose,
                                            out_3_face, #out_3_pose
                                            ])
            else:
                revised_model = Model(inp, [out_facial, out_1, out_2, out_3])

        return revised_model

    def inception_v1_single_task(self, tensor, pose, test):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)
        mobilenet_model.layers.pop()

        inp = mobilenet_model.input

        '''------------------------------------'''
        '''residual network_branch_1'''
        '''     block_1  '''
        r1_block_1_project_BN = mobilenet_model.get_layer('block_1_project_BN').output  # 56, 56, 24
        r1_dasm_1_conv_2d = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                   use_bias=False,
                                   name='r1_dasm_1_conv_2d', kernel_initializer='normal')(r1_block_1_project_BN)
        r1_dasm_1_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                          name='r1_dasm_1_BN')(r1_dasm_1_conv_2d)  # 28, 28, 32
        # r1_dasm_1_relu = ReLU(6., name='r1_dasm_1_relu')(r1_dasm_1_BN)

        block_3_project_BN = mobilenet_model.get_layer('block_3_project_BN').output  # 28, 28, 32
        # r1_dasm_combined_1_and_3_layer = keras.layers.add([r1_dasm_1_BN, block_3_project_BN])  # 28, 28, 32
        r1_dasm_combined_1_and_3_layer = Concatenate()([r1_dasm_1_BN, block_3_project_BN])  # 28, 28, 32

        '''{ first output '''
        out_1 = ReLU(6., name='out_1_relu')(r1_block_1_project_BN)
        out_1 = GlobalAveragePooling2D(name='out_1_GlobalAveragePooling2D')(out_1)
        out_1 = Dropout(rate=0.3)(out_1)
        out_1 = Dense(LearningConfig.landmark_len, name='out_1_dens', kernel_initializer='he_uniform')(out_1)
        '''first output }'''

        '''{ second output '''
        out_2 = ReLU(6., name='out_2_relu')(r1_dasm_combined_1_and_3_layer)
        out_2 = GlobalAveragePooling2D(name='out_2_GlobalAveragePooling2D')(out_2)
        out_2 = Dropout(rate=0.3)(out_2)
        out_2 = Dense(LearningConfig.landmark_len, name='out_2_dens', kernel_initializer='he_uniform')(out_2)
        '''second output }'''

        '''     block_2 '''
        x = Conv2D(6 * 32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_and_3_layer_expand')(r1_dasm_combined_1_and_3_layer)  # 28, 28, 6*32=192
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r1_dasm_combined_1_and_3_layer_expand_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r1_dasm_combined_1_and_3_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_and_3_layer_depthwise_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_depthwise_BN_relu')(x)

        x = Conv2D(32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_and_3_layer_project')(x)  # 28, 28, 6*32=192
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_and_3_layer_project_project_BN')(x)  # 28, 28, 32

        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_project_project_BN_relu')(x)

        y = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                   use_bias=False,
                   name='r1_dasm_1_and_3_reduce_conv_2d', kernel_initializer='normal')(x)  # 14, 14, 64
        r1_dasm_1_and_3_reduce_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                                       name='r1_dasm_1_and_3_reduce_BN')(y)
        # r1_dasm_1_and_3_reduce_BN_relu = ReLU(6., name='r1_dasm_1_and_3_reduce_BN_relu')(y)  # 14, 14, 64

        block_6_project_BN = mobilenet_model.get_layer('block_6_project_BN').output  # 14, 14, 64
        dasm_combined_1_3_and_6_layer = Concatenate()([block_6_project_BN, r1_dasm_1_and_3_reduce_BN])  # 14, 14, 64

        '''{ third output '''
        out_3 = ReLU(6., name='out_3_relu')(dasm_combined_1_3_and_6_layer)
        out_3 = GlobalAveragePooling2D(name='out_3_GlobalAveragePooling2D')(out_3)
        out_3 = Dropout(rate=0.3)(out_3)
        out_3 = Dense(LearningConfig.landmark_len, name='out_3_dens', kernel_initializer='he_uniform')(out_3)
        '''third output }'''


        '''     block_3 '''
        x = Conv2D(6 * 64, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_3_and_6_layer_expand')(dasm_combined_1_3_and_6_layer)  # 14, 14, 6*64=384
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r1_dasm_combined_1_3_and_6_layer_expand_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_3_and_6_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r1_dasm_combined_1_3_and_6_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_3_and_6_layer_depthwise_BN')(x)  # 14, 14, 64
        x = ReLU(6., name='r1_dasm_combined_1_3_and_6_layer_depthwise_relu')(x)
        x = Conv2D(64, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_3_and_6_layer_conv2d')(x)  # 14, 14, 64
        r1_dasm_combined_1_3_and_6_layer_expand_BN_expand_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                                                                  name='r1_dasm_combined_1_3_and_6_layer_conv2d_BN')(
            x)  # 14, 14, 64
        '''------------------------------------'''

        # res 3
        block_9_project_BN = mobilenet_model.get_layer('block_9_project_BN').output  # 14, 14, 64
        dasm_1_3_and_dasm_1_3_6_and_9_layer = Concatenate()([block_9_project_BN,
                                                             r1_dasm_combined_1_3_and_6_layer_expand_BN_expand_BN])  # 14, 14, 64

        '''{ forth output '''
        out_4 = ReLU(6., name='out_4_relu')(dasm_1_3_and_dasm_1_3_6_and_9_layer)
        out_4 = GlobalAveragePooling2D(name='out_4_GlobalAveragePooling2D')(out_4)
        out_4 = Dropout(rate=0.3)(out_4)
        out_4 = Dense(LearningConfig.landmark_len, name='out_4_dens', kernel_initializer='he_uniform')(out_4)
        '''forth output }'''

        '''reducing to 7'''
        Conv_0 = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same',
                        use_bias=False,
                        name='Conv_0', kernel_initializer='normal')(dasm_1_3_and_dasm_1_3_6_and_9_layer)  # 7, 7, 128
        Conv_0_BN = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_0_BN')(Conv_0)  # 14, 14, 64
        Conv_0_ReLU = ReLU(6., name='Conv_0_ReLU')(Conv_0_BN)

        x = Conv2D(256, kernel_size=1, use_bias=False, name='Conv_1')(Conv_0_ReLU)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
        x = ReLU(6., name='out_relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, name='dense_layer_out_2', activation='relu', kernel_initializer='he_uniform')(x)
        x = Dense(136, name='dense_layer_out_3', activation='relu', kernel_initializer='he_uniform')(x)
        x = Dense(136, name='dense_layer_out_4', activation='relu', kernel_initializer='he_uniform')(x)
        x = Dropout(rate=0.3)(x)
        out_facial = Dense(LearningConfig.landmark_len, name='out_asm')(x)  # data become between -1, 1

        out_pose = Dense(InputDataSize.pose_len, name='out_pose')(x)

        if pose:
            # revised_model = Model(inp, [asm_activation, Logits_nose, Logits_eyes, Logits_face, Logits_mouth, out_pose])
            revised_model = Model(inp, [out_facial, out_pose, out_1, out_2, out_3, out_4])
        else:
            # revised_model = Model(inp, [asm_activation, Logits_nose, Logits_eyes, Logits_face, Logits_mouth])
            revised_model = Model(inp, [out_facial, out_1, out_2, out_3, out_4])

        return revised_model

    def shrink_v6_mobileNet_v2_single_task(self, tensor, pose, test):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)

        mobilenet_model.layers.pop()

        inp = mobilenet_model.input

        '''------------------------------------'''
        '''residual network_branch_1'''
        '''     block_1  '''
        r1_block_1_project_BN = mobilenet_model.get_layer('block_1_project_BN').output  # 56, 56, 24
        r1_dasm_1_conv_2d = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                   use_bias=False,
                                   name='r1_dasm_1_conv_2d', kernel_initializer='normal')(r1_block_1_project_BN)
        r1_dasm_1_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                          name='r1_dasm_1_BN')(r1_dasm_1_conv_2d)  # 28, 28, 32
        # r1_dasm_1_relu = ReLU(6., name='r1_dasm_1_relu')(r1_dasm_1_BN)

        block_3_project_BN = mobilenet_model.get_layer('block_3_project_BN').output  # 28, 28, 32
        r1_dasm_combined_1_and_3_layer = Concatenate()([r1_dasm_1_BN, block_3_project_BN])  # 28, 28, 32

        '''     block_2 '''
        x = Conv2D(6 * 32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_and_3_layer_expand')(r1_dasm_combined_1_and_3_layer)  # 28, 28, 6*32=192
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r1_dasm_combined_1_and_3_layer_expand_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r1_dasm_combined_1_and_3_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_and_3_layer_depthwise_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_depthwise_BN_relu')(x)

        x = Conv2D(32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_and_3_layer_project')(x)  # 28, 28, 6*32=192
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_and_3_layer_project_project_BN')(x)  # 28, 28, 32

        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_project_project_BN_relu')(x)

        y = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                   use_bias=False,
                   name='r1_dasm_1_and_3_reduce_conv_2d', kernel_initializer='normal')(x)  # 14, 14, 64
        r1_dasm_1_and_3_reduce_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                                       name='r1_dasm_1_and_3_reduce_BN')(y)
        # r1_dasm_1_and_3_reduce_BN_relu = ReLU(6., name='r1_dasm_1_and_3_reduce_BN_relu')(y)  # 14, 14, 64

        block_6_project_BN = mobilenet_model.get_layer('block_6_project_BN').output  # 14, 14, 64
        dasm_combined_1_3_and_6_layer = Concatenate()([block_6_project_BN,r1_dasm_1_and_3_reduce_BN])  # 14, 14, 64
        '''     block_3 '''
        x = Conv2D(6 * 64, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_3_and_6_layer_expand')(dasm_combined_1_3_and_6_layer)  # 14, 14, 6*64=384
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r1_dasm_combined_1_3_and_6_layer_expand_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_3_and_6_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r1_dasm_combined_1_3_and_6_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_3_and_6_layer_depthwise_BN')(x)  # 14, 14, 64
        x = ReLU(6., name='r1_dasm_combined_1_3_and_6_layer_depthwise_relu')(x)
        x = Conv2D(64, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_3_and_6_layer_conv2d')(x)  # 14, 14, 64
        r1_dasm_combined_1_3_and_6_layer_expand_BN_expand_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                                                                  name='r1_dasm_combined_1_3_and_6_layer_conv2d_BN')(
            x)  # 14, 14, 64
        '''------------------------------------'''

        # res 3
        block_9_project_BN = mobilenet_model.get_layer('block_9_project_BN').output  # 14, 14, 64
        dasm_1_3_and_dasm_1_3_6_and_9_layer = Concatenate()([block_9_project_BN,
                                                                r1_dasm_combined_1_3_and_6_layer_expand_BN_expand_BN])  # 14, 14, 64

        '''reducing to 7'''
        Conv_0 = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same',
                        use_bias=False,
                        name='Conv_0', kernel_initializer='normal')(dasm_1_3_and_dasm_1_3_6_and_9_layer)  # 7, 7, 128
        Conv_0_BN = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_0_BN')(Conv_0)  # 14, 14, 64
        Conv_0_ReLU = ReLU(6., name='Conv_0_ReLU')(Conv_0_BN)

        x = Conv2D(256, kernel_size=1, use_bias=False, name='Conv_1')(Conv_0_ReLU)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
        x = ReLU(6., name='out_relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, name='dense_layer_out_2', activation='relu', kernel_initializer='he_uniform')(x)

        # custom activation:
        Logits_asm = Dense(LearningConfig.landmark_len, activation='tanh', name='out_asm')(
            x)  # data become between -1, 1

        # tanh_activation = Activation('tanh', name='tanh_activation')(Logits_asm)  # data become between -1, 1

        if test:  # use batch size 1
            asm_activation = Activation(self.custom_activation_test, name='asm_activation')(Logits_asm)
        else:  # use batch size real
            asm_activation = Activation(self.custom_activation, name='asm_activation')(Logits_asm)

        # multitask output layers
        # Logits_face = Dense(InputDataSize.landmark_face_len, name='out_face')(x)
        # Logits_nose = Dense(InputDataSize.landmark_nose_len, name='out_nose')(x)
        # Logits_eyes = Dense(InputDataSize.landmark_eys_len, name='out_eyes')(x)
        # Logits_mouth = Dense(InputDataSize.landmark_mouth_len, name='out_mouth')(x)

        out_pose = Dense(InputDataSize.pose_len, name='out_pose')(x)

        if pose:
            # revised_model = Model(inp, [asm_activation, Logits_nose, Logits_eyes, Logits_face, Logits_mouth, out_pose])
            revised_model = Model(inp, [asm_activation, out_pose])
        else:
            # revised_model = Model(inp, [asm_activation, Logits_nose, Logits_eyes, Logits_face, Logits_mouth])
            revised_model = Model(inp, [asm_activation])

        return revised_model

    def shrink_v6_mobileNet_v2_multi_task(self, tensor, pose, test):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)

        mobilenet_model.layers.pop()

        inp = mobilenet_model.input

        '''------------------------------------'''
        '''residual network_branch_1'''
        '''     block_1  '''
        r1_block_1_project_BN = mobilenet_model.get_layer('block_1_project_BN').output  # 56, 56, 24
        r1_dasm_1_conv_2d = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                   use_bias=False,
                                   name='r1_dasm_1_conv_2d', kernel_initializer='normal')(r1_block_1_project_BN)
        r1_dasm_1_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                          name='r1_dasm_1_BN')(r1_dasm_1_conv_2d)  # 28, 28, 32
        # r1_dasm_1_relu = ReLU(6., name='r1_dasm_1_relu')(r1_dasm_1_BN)

        block_3_project_BN = mobilenet_model.get_layer('block_3_project_BN').output  # 28, 28, 32
        r1_dasm_combined_1_and_3_layer = Concatenate()([r1_dasm_1_BN, block_3_project_BN])  # 28, 28, 32

        '''     block_2 '''
        x = Conv2D(6 * 32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_and_3_layer_expand')(r1_dasm_combined_1_and_3_layer)  # 28, 28, 6*32=192
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r1_dasm_combined_1_and_3_layer_expand_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r1_dasm_combined_1_and_3_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_and_3_layer_depthwise_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_depthwise_BN_relu')(x)

        x = Conv2D(32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_and_3_layer_project')(x)  # 28, 28, 6*32=192
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_and_3_layer_project_project_BN')(x)  # 28, 28, 32

        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_project_project_BN_relu')(x)

        y = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                   use_bias=False,
                   name='r1_dasm_1_and_3_reduce_conv_2d', kernel_initializer='normal')(x)  # 14, 14, 64
        r1_dasm_1_and_3_reduce_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                                       name='r1_dasm_1_and_3_reduce_BN')(y)
        # r1_dasm_1_and_3_reduce_BN_relu = ReLU(6., name='r1_dasm_1_and_3_reduce_BN_relu')(y)  # 14, 14, 64

        block_6_project_BN = mobilenet_model.get_layer('block_6_project_BN').output  # 14, 14, 64
        dasm_combined_1_3_and_6_layer = Concatenate()([block_6_project_BN,r1_dasm_1_and_3_reduce_BN])  # 14, 14, 64
        '''     block_3 '''
        x = Conv2D(6 * 64, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_3_and_6_layer_expand')(dasm_combined_1_3_and_6_layer)  # 14, 14, 6*64=384
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r1_dasm_combined_1_3_and_6_layer_expand_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_3_and_6_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r1_dasm_combined_1_3_and_6_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_3_and_6_layer_depthwise_BN')(x)  # 14, 14, 64
        x = ReLU(6., name='r1_dasm_combined_1_3_and_6_layer_depthwise_relu')(x)
        x = Conv2D(64, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_3_and_6_layer_conv2d')(x)  # 14, 14, 64
        r1_dasm_combined_1_3_and_6_layer_expand_BN_expand_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                                                                  name='r1_dasm_combined_1_3_and_6_layer_conv2d_BN')(
            x)  # 14, 14, 64
        '''------------------------------------'''

        # res 3
        block_9_project_BN = mobilenet_model.get_layer('block_9_project_BN').output  # 14, 14, 64
        dasm_1_3_and_dasm_1_3_6_and_9_layer = Concatenate()([block_9_project_BN,
                                                                r1_dasm_combined_1_3_and_6_layer_expand_BN_expand_BN])  # 14, 14, 64

        '''reducing to 7'''
        Conv_0 = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same',
                        use_bias=False,
                        name='Conv_0', kernel_initializer='normal')(dasm_1_3_and_dasm_1_3_6_and_9_layer)  # 7, 7, 128
        Conv_0_BN = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_0_BN')(Conv_0)  # 14, 14, 64
        Conv_0_ReLU = ReLU(6., name='Conv_0_ReLU')(Conv_0_BN)

        x = Conv2D(256, kernel_size=1, use_bias=False, name='Conv_1')(Conv_0_ReLU)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
        x = ReLU(6., name='out_relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, name='dense_layer_out_2', activation='relu', kernel_initializer='he_uniform')(x)

        # custom activation:
        Logits_asm = Dense(LearningConfig.landmark_len, activation='tanh', name='out_asm')(
            x)  # data become between -1, 1

        # tanh_activation = Activation('tanh', name='tanh_activation')(Logits_asm)  # data become between -1, 1

        if test:  # use batch size 1
            asm_activation = Activation(self.custom_activation_test, name='asm_activation')(Logits_asm)
        else:  # use batch size real
            asm_activation = Activation(self.custom_activation, name='asm_activation')(Logits_asm)

        # multitask output layers
        Logits_face = Dense(InputDataSize.landmark_face_len, name='out_face')(x)
        Logits_nose = Dense(InputDataSize.landmark_nose_len, name='out_nose')(x)
        Logits_eyes = Dense(InputDataSize.landmark_eys_len, name='out_eyes')(x)
        Logits_mouth = Dense(InputDataSize.landmark_mouth_len, name='out_mouth')(x)

        out_pose = Dense(InputDataSize.pose_len, name='out_pose')(x)

        if pose:
            revised_model = Model(inp, [asm_activation, Logits_nose, Logits_eyes, Logits_face, Logits_mouth, out_pose])
        else:
            revised_model = Model(inp, [asm_activation, Logits_nose, Logits_eyes, Logits_face, Logits_mouth])

        return revised_model

    def shrink_v5_mobileNet_v2_multi_task(self, tensor, pose, test):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)

        mobilenet_model.layers.pop()

        inp = mobilenet_model.input

        '''------------------------------------'''
        '''residual network_branch_1'''
        '''     block_1  '''
        r1_block_1_project_BN = mobilenet_model.get_layer('block_1_project_BN').output  # 56, 56, 24
        r1_dasm_1_conv_2d = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                   use_bias=False,
                                   name='r1_dasm_1_conv_2d', kernel_initializer='normal')(r1_block_1_project_BN)
        r1_dasm_1_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                          name='r1_dasm_1_BN')(r1_dasm_1_conv_2d)  # 28, 28, 32
        # r1_dasm_1_relu = ReLU(6., name='r1_dasm_1_relu')(r1_dasm_1_BN)

        block_3_project_BN = mobilenet_model.get_layer('block_3_project_BN').output  # 28, 28, 32
        r1_dasm_combined_1_and_3_layer = Concatenate()([r1_dasm_1_BN, block_3_project_BN])  # 28, 28, 32

        '''     block_2 '''
        x = Conv2D(6 * 32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_and_3_layer_expand')(r1_dasm_combined_1_and_3_layer)  # 28, 28, 6*32=192
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r1_dasm_combined_1_and_3_layer_expand_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r1_dasm_combined_1_and_3_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_and_3_layer_depthwise_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_depthwise_BN_relu')(x)

        x = Conv2D(32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_and_3_layer_project')(x)  # 28, 28, 6*32=192
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_and_3_layer_project_project_BN')(x)  # 28, 28, 32

        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_project_project_BN_relu')(x)

        y = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                   use_bias=False,
                   name='r1_dasm_1_and_3_reduce_conv_2d', kernel_initializer='normal')(x)  # 14, 14, 64
        r1_dasm_1_and_3_reduce_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                                       name='r1_dasm_1_and_3_reduce_BN')(y)
        # r1_dasm_1_and_3_reduce_BN_relu = ReLU(6., name='r1_dasm_1_and_3_reduce_BN_relu')(y)  # 14, 14, 64

        block_6_project_BN = mobilenet_model.get_layer('block_6_project_BN').output  # 14, 14, 64
        dasm_combined_1_3_and_6_layer = Concatenate()([block_6_project_BN,r1_dasm_1_and_3_reduce_BN])  # 14, 14, 64
        '''     block_3 '''
        x = Conv2D(6 * 64, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_3_and_6_layer_expand')(dasm_combined_1_3_and_6_layer)  # 14, 14, 6*64=384
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r1_dasm_combined_1_3_and_6_layer_expand_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_3_and_6_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r1_dasm_combined_1_3_and_6_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_3_and_6_layer_depthwise_BN')(x)  # 14, 14, 64
        x = ReLU(6., name='r1_dasm_combined_1_3_and_6_layer_depthwise_relu')(x)
        x = Conv2D(64, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_3_and_6_layer_conv2d')(x)  # 14, 14, 64
        r1_dasm_combined_1_3_and_6_layer_expand_BN_expand_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                                                                  name='r1_dasm_combined_1_3_and_6_layer_conv2d_BN')(
            x)  # 14, 14, 64
        '''------------------------------------'''
        '''residual network_branch_2'''
        # res_line_2
        block_1_project_BN = mobilenet_model.get_layer('block_1_project_BN').output  # 56, 56, 24
        dasm_1_conv_2d = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                use_bias=False,
                                name='r2_dasm_1_conv_2d', kernel_initializer='normal')(block_1_project_BN)

        dasm_1_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                       name='r2_dasm_1_BN')(dasm_1_conv_2d)  # 28, 28, 32

        block_3_project_BN = mobilenet_model.get_layer('block_3_project_BN').output  # 28, 28, 32
        dasm_combined_1_and_3_layer = Concatenate()([dasm_1_BN, block_3_project_BN])  # 28, 28, 32

        # rev_res 1
        x = Conv2D(6 * 32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r2_dasm_combined_1_and_3_layer_expand')(dasm_combined_1_and_3_layer)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r2_dasm_combined_1_and_3_layer_expand_BN')(x)
        x = ReLU(6., name='r2_dasm_combined_1_and_3_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r2_dasm_combined_1_and_3_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r2_dasm_combined_1_and_3_layer_depthwise_BN')(x)
        x = ReLU(6., name='r2_dasm_combined_1_and_3_layer_depthwise_BN_relu')(x)
        x = Conv2D(32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r2_dasm_combined_1_and_3_layer_project')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='dasm_combined_1_and_3_layer_project_project_BN')(x)  # 28, 28, 32
        x = ReLU(6., name='r2_dasm_combined_1_and_3_layer_project_project_BN_relu')(x)
        y = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                   use_bias=False,
                   name='r2_dasm_1_and_3_reduce_conv_2d', kernel_initializer='normal')(x)
        dasm_1_and_3_reduce_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                                    name='r2_dasm_1_and_3_reduce_BN')(y)

        # res 3
        block_9_project_BN = mobilenet_model.get_layer('block_9_project_BN').output  # 14, 14, 64
        dasm_1_3_and_dasm_1_3_6_and_9_layer = Concatenate()([block_9_project_BN,
                                                                dasm_1_and_3_reduce_BN,
                                                                r1_dasm_combined_1_3_and_6_layer_expand_BN_expand_BN])  # 14, 14, 64

        '''reducing to 7'''
        Conv_0 = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same',
                        use_bias=False,
                        name='Conv_0', kernel_initializer='normal')(dasm_1_3_and_dasm_1_3_6_and_9_layer)  # 7, 7, 128
        Conv_0_BN = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_0_BN')(Conv_0)  # 14, 14, 64
        Conv_0_ReLU = ReLU(6., name='Conv_0_ReLU')(Conv_0_BN)

        x = Conv2D(256, kernel_size=1, use_bias=False, name='Conv_1')(Conv_0_ReLU)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
        x = ReLU(6., name='out_relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, name='dense_layer_out_2', activation='relu', kernel_initializer='he_uniform')(x)

        # custom activation:
        Logits_asm = Dense(LearningConfig.landmark_len, activation='tanh', name='out_asm')(
            x)  # data become between -1, 1

        # tanh_activation = Activation('tanh', name='tanh_activation')(Logits_asm)  # data become between -1, 1

        if test:  # use batch size 1
            asm_activation = Activation(self.custom_activation_test, name='asm_activation')(Logits_asm)
        else:  # use batch size real
            asm_activation = Activation(self.custom_activation, name='asm_activation')(Logits_asm)

        # multitask output layers
        Logits_face = Dense(InputDataSize.landmark_face_len, name='out_face')(x)
        Logits_nose = Dense(InputDataSize.landmark_nose_len, name='out_nose')(x)
        Logits_eyes = Dense(InputDataSize.landmark_eys_len, name='out_eyes')(x)
        Logits_mouth = Dense(InputDataSize.landmark_mouth_len, name='out_mouth')(x)

        out_pose = Dense(InputDataSize.pose_len, name='out_pose')(x)

        if pose:
            revised_model = Model(inp, [asm_activation, Logits_nose, Logits_eyes, Logits_face, Logits_mouth, out_pose])
        else:
            revised_model = Model(inp, [asm_activation, Logits_nose, Logits_eyes, Logits_face, Logits_mouth])
        return revised_model

    def shrink_v4_mobileNet_v2_multi_task(self, tensor, pose, test):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)

        mobilenet_model.layers.pop()

        inp = mobilenet_model.input

        '''------------------------------------'''
        '''residual network_branch_1'''
        '''     block_1  '''
        r1_block_1_project_BN = mobilenet_model.get_layer('block_1_project_BN').output  # 56, 56, 24
        r1_dasm_1_conv_2d = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                   use_bias=False,
                                   name='r1_dasm_1_conv_2d', kernel_initializer='normal')(r1_block_1_project_BN)
        r1_dasm_1_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                          name='r1_dasm_1_BN')(r1_dasm_1_conv_2d)  # 28, 28, 32
        # r1_dasm_1_relu = ReLU(6., name='r1_dasm_1_relu')(r1_dasm_1_BN)

        block_3_project_BN = mobilenet_model.get_layer('block_3_project_BN').output  # 28, 28, 32
        r1_dasm_combined_1_and_3_layer = keras.layers.add([r1_dasm_1_BN, block_3_project_BN])  # 28, 28, 32

        '''     block_2 '''
        x = Conv2D(6 * 32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_and_3_layer_expand')(r1_dasm_combined_1_and_3_layer)  # 28, 28, 6*32=192
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r1_dasm_combined_1_and_3_layer_expand_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r1_dasm_combined_1_and_3_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_and_3_layer_depthwise_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_depthwise_BN_relu')(x)

        x = Conv2D(32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_and_3_layer_project')(x)  # 28, 28, 6*32=192
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_and_3_layer_project_project_BN')(x)  # 28, 28, 32

        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_project_project_BN_relu')(x)

        y = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                   use_bias=False,
                   name='r1_dasm_1_and_3_reduce_conv_2d', kernel_initializer='normal')(x)  # 14, 14, 64
        r1_dasm_1_and_3_reduce_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                                       name='r1_dasm_1_and_3_reduce_BN')(y)
        # r1_dasm_1_and_3_reduce_BN_relu = ReLU(6., name='r1_dasm_1_and_3_reduce_BN_relu')(y)  # 14, 14, 64

        block_6_project_BN = mobilenet_model.get_layer('block_6_project_BN').output  # 14, 14, 64
        dasm_combined_1_3_and_6_layer = keras.layers.add([block_6_project_BN,
                                                          r1_dasm_1_and_3_reduce_BN])  # 14, 14, 64
        '''     block_3 '''
        x = Conv2D(6 * 64, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_3_and_6_layer_expand')(dasm_combined_1_3_and_6_layer)  # 14, 14, 6*64=384
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r1_dasm_combined_1_3_and_6_layer_expand_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_3_and_6_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r1_dasm_combined_1_3_and_6_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_3_and_6_layer_depthwise_BN')(x)  # 14, 14, 64
        x = ReLU(6., name='r1_dasm_combined_1_3_and_6_layer_depthwise_relu')(x)
        x = Conv2D(64, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_3_and_6_layer_conv2d')(x)  # 14, 14, 64
        r1_dasm_combined_1_3_and_6_layer_expand_BN_expand_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                                                                  name='r1_dasm_combined_1_3_and_6_layer_conv2d_BN')(
            x)  # 14, 14, 64
        '''------------------------------------'''
        '''residual network_branch_2'''
        # res_line_2
        block_1_project_BN = mobilenet_model.get_layer('block_1_project_BN').output  # 56, 56, 24
        dasm_1_conv_2d = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                use_bias=False,
                                name='r2_dasm_1_conv_2d', kernel_initializer='normal')(block_1_project_BN)

        dasm_1_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                       name='r2_dasm_1_BN')(dasm_1_conv_2d)  # 28, 28, 32

        # dasm_1_relu = ReLU(6., name='r2_dasm_1_relu')(dasm_1_BN)

        block_3_project_BN = mobilenet_model.get_layer('block_3_project_BN').output  # 28, 28, 32
        dasm_combined_1_and_3_layer = keras.layers.add([dasm_1_BN, block_3_project_BN])  # 28, 28, 32

        # rev_res 1
        x = Conv2D(6 * 32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r2_dasm_combined_1_and_3_layer_expand')(dasm_combined_1_and_3_layer)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r2_dasm_combined_1_and_3_layer_expand_BN')(x)
        x = ReLU(6., name='r2_dasm_combined_1_and_3_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r2_dasm_combined_1_and_3_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r2_dasm_combined_1_and_3_layer_depthwise_BN')(x)
        x = ReLU(6., name='r2_dasm_combined_1_and_3_layer_depthwise_BN_relu')(x)
        x = Conv2D(32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r2_dasm_combined_1_and_3_layer_project')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='dasm_combined_1_and_3_layer_project_project_BN')(x)  # 28, 28, 32
        x = ReLU(6., name='r2_dasm_combined_1_and_3_layer_project_project_BN_relu')(x)
        y = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                   use_bias=False,
                   name='r2_dasm_1_and_3_reduce_conv_2d', kernel_initializer='normal')(x)
        dasm_1_and_3_reduce_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                                    name='r2_dasm_1_and_3_reduce_BN')(y)

        # res 3
        block_9_project_BN = mobilenet_model.get_layer('block_9_project_BN').output  # 14, 14, 64
        dasm_1_3_and_dasm_1_3_6_and_9_layer = keras.layers.add([block_9_project_BN,
                                                                dasm_1_and_3_reduce_BN,
                                                                r1_dasm_combined_1_3_and_6_layer_expand_BN_expand_BN])  # 14, 14, 64

        '''reducing to 7'''
        Conv_0 = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same',
                   use_bias=False,
                   name='Conv_0', kernel_initializer='normal')(dasm_1_3_and_dasm_1_3_6_and_9_layer)  # 7, 7, 128
        Conv_0_BN = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_0_BN')(Conv_0)  # 14, 14, 64
        Conv_0_ReLU = ReLU(6., name='Conv_0_ReLU')(Conv_0_BN)

        x = Conv2D(256, kernel_size=1, use_bias=False, name='Conv_1')(Conv_0_ReLU)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
        x = ReLU(6., name='out_relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, name='dense_layer_out_2', activation='relu', kernel_initializer='he_uniform')(x)

        # custom activation
        Logits_asm = Dense(LearningConfig.landmark_len,  activation='tanh', name='out_asm')(x)  # data become between -1, 1

        # tanh_activation = Activation('tanh', name='tanh_activation')(Logits_asm)  # data become between -1, 1

        if test:  # use batch size 1
            asm_activation = Activation(self.custom_activation_test, name='asm_activation')(Logits_asm)
        else:  # use batch size real
            asm_activation = Activation(self.custom_activation, name='asm_activation')(Logits_asm)

        # multitask output layers
        Logits_face = Dense(InputDataSize.landmark_face_len, name='out_face')(x)
        Logits_nose = Dense(InputDataSize.landmark_nose_len, name='out_nose')(x)
        Logits_eyes = Dense(InputDataSize.landmark_eys_len, name='out_eyes')(x)
        Logits_mouth = Dense(InputDataSize.landmark_mouth_len, name='out_mouth')(x)

        out_pose = Dense(InputDataSize.pose_len, name='out_pose')(x)

        if pose:
            revised_model = Model(inp, [asm_activation, Logits_nose, Logits_eyes, Logits_face, Logits_mouth, out_pose])
        else:
            revised_model = Model(inp, [asm_activation, Logits_nose, Logits_eyes, Logits_face, Logits_mouth])
        return revised_model

    def shrink_v2_mobileNet_v2_multi_task(self, tensor, pose):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)

        mobilenet_model.layers.pop()

        inp = mobilenet_model.input

        '''------------------------------------'''
        '''residual network_branch_1'''
        '''     block_1  '''
        r1_block_1_project_BN = mobilenet_model.get_layer('block_1_project_BN').output  # 56, 56, 24
        r1_dasm_1_conv_2d = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                   use_bias=False,
                                   name='r1_dasm_1_conv_2d', kernel_initializer='normal')(r1_block_1_project_BN)
        r1_dasm_1_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                          name='r1_dasm_1_BN')(r1_dasm_1_conv_2d)  # 28, 28, 32

        block_3_project_BN = mobilenet_model.get_layer('block_3_project_BN').output  # 28, 28, 32
        r1_dasm_combined_1_and_3_layer = keras.layers.add([r1_dasm_1_BN, block_3_project_BN])  # 28, 28, 32

        '''     block_2 '''
        x = Conv2D(6 * 32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_and_3_layer_expand')(r1_dasm_combined_1_and_3_layer)  # 28, 28, 6*32=192
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r1_dasm_combined_1_and_3_layer_expand_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r1_dasm_combined_1_and_3_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_and_3_layer_depthwise_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_depthwise_BN_relu')(x)

        x = Conv2D(32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_and_3_layer_project')(x)  # 28, 28, 6*32=192
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_and_3_layer_project_project_BN')(x)  # 28, 28, 32

        x = ReLU(6., name='r1_dasm_combined_1_and_3_layer_project_project_BN_relu')(x)

        y = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                   use_bias=False,
                   name='r1_dasm_1_and_3_reduce_conv_2d', kernel_initializer='normal')(x)  # 14, 14, 64
        r1_dasm_1_and_3_reduce_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                                       name='r1_dasm_1_and_3_reduce_BN')(y)
        # r1_dasm_1_and_3_reduce_BN_relu = ReLU(6., name='r1_dasm_1_and_3_reduce_BN_relu')(y)  # 14, 14, 64

        block_6_project_BN = mobilenet_model.get_layer('block_6_project_BN').output  # 14, 14, 64
        dasm_combined_1_3_and_6_layer = keras.layers.add([block_6_project_BN,
                                                          r1_dasm_1_and_3_reduce_BN])  # 14, 14, 64
        '''     block_3 '''
        x = Conv2D(6 * 64, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_3_and_6_layer_expand')(dasm_combined_1_3_and_6_layer)  # 14, 14, 6*64=384
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r1_dasm_combined_1_3_and_6_layer_expand_BN')(x)
        x = ReLU(6., name='r1_dasm_combined_1_3_and_6_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r1_dasm_combined_1_3_and_6_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r1_dasm_combined_1_3_and_6_layer_depthwise_BN')(x)  # 14, 14, 64
        x = ReLU(6., name='r1_dasm_combined_1_3_and_6_layer_depthwise_relu')(x)
        x = Conv2D(64, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r1_dasm_combined_1_3_and_6_layer_conv2d')(x)  # 14, 14, 64
        r1_dasm_combined_1_3_and_6_layer_expand_BN_expand_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                                                                  name='r1_dasm_combined_1_3_and_6_layer_conv2d_BN')(
            x)  # 14, 14, 64
        '''------------------------------------'''
        '''residual network_branch_2'''
        # res_line_2
        block_1_project_BN = mobilenet_model.get_layer('block_1_project_BN').output  # 56, 56, 24
        dasm_1_conv_2d = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                use_bias=False,
                                name='r2_dasm_1_conv_2d', kernel_initializer='normal')(block_1_project_BN)
        dasm_1_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                       name='r2_dasm_1_BN')(dasm_1_conv_2d)  # 28, 28, 32
        # dasm_1_relu = ReLU(6., name='r2_dasm_1_relu')(dasm_1_BN)

        block_3_project_BN = mobilenet_model.get_layer('block_3_project_BN').output  # 28, 28, 32
        dasm_combined_1_and_3_layer = keras.layers.add([dasm_1_BN, block_3_project_BN])  # 28, 28, 32

        # rev_res 1
        x = Conv2D(6 * 32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r2_dasm_combined_1_and_3_layer_expand')(dasm_combined_1_and_3_layer)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='r2_dasm_combined_1_and_3_layer_expand_BN')(x)
        x = ReLU(6., name='r2_dasm_combined_1_and_3_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='r2_dasm_combined_1_and_3_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='r2_dasm_combined_1_and_3_layer_depthwise_BN')(x)
        x = ReLU(6., name='r2_dasm_combined_1_and_3_layer_depthwise_BN_relu')(x)
        x = Conv2D(32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='r2_dasm_combined_1_and_3_layer_project')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='dasm_combined_1_and_3_layer_project_project_BN')(x)  # 28, 28, 32
        x = ReLU(6., name='r2_dasm_combined_1_and_3_layer_project_project_BN_relu')(x)
        y = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                   use_bias=False,
                   name='r2_dasm_1_and_3_reduce_conv_2d', kernel_initializer='normal')(x)
        dasm_1_and_3_reduce_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                                    name='r2_dasm_1_and_3_reduce_BN')(y)

        # res 3
        block_9_project_BN = mobilenet_model.get_layer('block_9_project_BN').output  # 14, 14, 64
        dasm_1_3_and_dasm_1_3_6_and_9_layer = keras.layers.add([block_9_project_BN,
                                                                dasm_1_and_3_reduce_BN,
                                                                r1_dasm_combined_1_3_and_6_layer_expand_BN_expand_BN])  # 14, 14, 64

        '''reducing to 7'''
        Conv_0 = Conv2D(128, kernel_size=1, padding='same', use_bias=False, activation=None,
                        name='Conv_0')(dasm_1_3_and_dasm_1_3_6_and_9_layer)  # 14, 14, 64
        Conv_0_BN = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_0_BN')(Conv_0)  # 14, 14, 64
        Conv_0_ReLU = ReLU(6., name='Conv_0_ReLU')(Conv_0_BN)
        # ..
        x = Conv2D(256, kernel_size=1, use_bias=False, name='Conv_1')(Conv_0_ReLU)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
        x = ReLU(6., name='out_relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(LearningConfig.landmark_len, name='dense_layer_out_2', activation='relu'
                  , kernel_initializer='he_uniform')(x)

        # multitask output layers
        # Logits_pose = Dense(InputDataSize.pose_len, name='out_pose')(x)
        Logits = Dense(LearningConfig.landmark_len, name='out_main')(x)
        Logits_face = Dense(InputDataSize.landmark_face_len, name='out_face')(x)
        Logits_nose = Dense(InputDataSize.landmark_nose_len, name='out_nose')(x)
        Logits_eyes = Dense(InputDataSize.landmark_eys_len, name='out_eyes')(x)
        Logits_mouth = Dense(InputDataSize.landmark_mouth_len, name='out_mouth')(x)

        out_pose = Dense(InputDataSize.pose_len, name='out_pose')(x)

        if pose:
            revised_model = Model(inp, [Logits, Logits_nose, Logits_eyes, Logits_face, Logits_mouth, out_pose])
        else:
            revised_model = Model(inp, [Logits, Logits_nose, Logits_eyes, Logits_face, Logits_mouth])

        return revised_model

    def shrink_mobileNet_v2(self, tensor):  # shallow
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)

        mobilenet_model.layers.pop()

        inp = mobilenet_model.input

        # res 1
        block_1_project_BN = mobilenet_model.get_layer('block_1_project_BN').output  # 56, 56, 24
        dasm_1_conv_2d = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                use_bias=False,
                                name='dasm_1_conv_2d', kernel_initializer='normal')(block_1_project_BN)
        dasm_1_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                       name='dasm_1_BN')(dasm_1_conv_2d)  # 28, 28, 32
        dasm_1_relu = ReLU(6., name='dasm_1_relu')(dasm_1_BN)

        block_3_project_BN = mobilenet_model.get_layer('block_3_project_BN').output  # 28, 28, 32
        dasm_combined_1_and_3_layer = keras.layers.add([dasm_1_relu, block_3_project_BN])  # 28, 28, 32

        # rev_res 1
        x = Conv2D(6 * 32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='dasm_combined_1_and_3_layer_expand')(dasm_combined_1_and_3_layer)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='dasm_combined_1_and_3_layer_expand_BN')(x)
        x = ReLU(6., name='dasm_combined_1_and_3_layer_expand_relu')(x)
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False, padding='same',
                            name='dasm_combined_1_and_3_layer_depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='dasm_combined_1_and_3_layer_depthwise_BN')(x)
        x = ReLU(6., name='dasm_combined_1_and_3_layer_depthwise_BN_relu')(x)
        x = Conv2D(32, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name='dasm_combined_1_and_3_layer_project')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='dasm_combined_1_and_3_layer_project_project_BN')(x)  # 28, 28, 32
        x = ReLU(6., name='dasm_combined_1_and_3_layer_project_project_BN_relu')(x)
        y = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                   use_bias=False,
                   name='dasm_1_and_3_reduce_conv_2d', kernel_initializer='normal')(x)
        dasm_1_and_3_reduce_BN = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='dasm_1_and_3_reduce_BN')(y)
        # res 3
        block_9_project_BN = mobilenet_model.get_layer('block_9_project_BN').output  # 14, 14, 96
        dasm_combined_1_3_and_9_layer = keras.layers.add([block_9_project_BN,
                                                          dasm_1_and_3_reduce_BN])  # 14, 14, 96

        x = Conv2D(128, kernel_size=1, use_bias=False, name='Conv_1')(dasm_combined_1_3_and_9_layer)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
        x = ReLU(6., name='out_relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(LearningConfig.landmark_len, name='dense_layer_out_2', activation='relu'
                  , kernel_initializer='he_uniform')(x)
        Logits = Dense(LearningConfig.landmark_len, name='out')(x)

        # revised_model = Model(inp, [out_main, out_res])
        revised_model = Model(inp, Logits)
        return revised_model

    def resNet_50_main(self, tensor):
        resnet_model = resnet50.ResNet50(include_top=True,
                                         weights=None,
                                         input_tensor=tensor,
                                         input_shape=[224, 224, 3],
                                         pooling=None)

        resnet_model.layers.pop()

        inp = resnet_model.input

        x = resnet_model.get_layer('avg_pool').output  # 2048

        x = Dense(LearningConfig.landmark_len + InputDataSize.pose_len, name='dense_1', activation='relu',
                  kernel_initializer='he_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)

        out_face = Dense(LearningConfig.landmark_len, name='out_face')(x)
        out_pose = Dense(InputDataSize.pose_len, name='out_pose')(x)

        revised_model = Model(inp, [out_face, out_pose])

        revised_model.summary()
        model_json = revised_model.to_json()

        with open("resnet_50_revised.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def resNet_50_main_inception_both_Discriminator(self, tensor):
        resnet_model = resnet50.ResNet50(include_top=True,
                                         weights=None,
                                         input_tensor=tensor,
                                         input_shape=[56, 56, 3],
                                         pooling=None)

        resnet_model.layers.pop()

        inp = resnet_model.input

        x = resnet_model.get_layer('avg_pool').output  # 2048

        x = Dense(LearningConfig.landmark_len + InputDataSize.pose_len, name='softmax', activation='softmax',
                  kernel_initializer='he_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)

        out_heatmap = Dense(2, name='out_hm')(x)
        out_heatmap_all = Dense(2, name='out_hm_all')(x)
        out_face = Dense(2, name='out_face')(x)
        out_pose = Dense(2, name='out_pose')(x)
        '''regression branch }'''
        '''} output '''

        revised_model = Model(inp, [
            out_heatmap, out_heatmap_all, out_face, out_pose,
        ])

        revised_model.summary()
        # plot_model(revised_model, to_file='resnet_50_incep_BOTH.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("resnet_50_incep_BOTH.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def mnv2_hm_r_v2(self, tensor, inception_mode=False):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)

        mobilenet_model.layers.pop()

        inp = mobilenet_model.input
        '''heatmap can not be generated from activation layers, so we use out_relu'''
        x = mobilenet_model.get_layer('out_relu').output  # 7, 7, 1280
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            kernel_initializer='he_uniform')(x)  # 14, 14, 256
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            kernel_initializer='he_uniform')(x)  # 28, 28, 256
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            kernel_initializer='he_uniform')(x)  # 56, 56, 256
        bn_0 = BatchNormalization(name='bn_0')(x)
        x = ReLU()(bn_0)

        '''reduce to  7'''
        x = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_uniform')(x)  # 28, 28, 256
        bn_1 = BatchNormalization(name='bn_1')(x)  # 28, 28, 256
        x = ReLU()(bn_1)

        x = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_uniform')(x)  # 14, 14, 256
        bn_2 = BatchNormalization(name='bn_2')(x)  # 14, 14, 256
        x = ReLU()(bn_2)

        x = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_uniform')(x)  # 7, 7 , 256
        bn_3 = BatchNormalization(name='bn_3')(x)  # 7, 7 , 256
        x = ReLU()(bn_3)

        '''increase to  56'''
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            name='deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
        x = BatchNormalization()(x)
        x = keras.layers.add([x, bn_2])  # 14, 14, 256
        x = ReLU()(x)

        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            name='deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
        x = BatchNormalization()(x)
        x = keras.layers.add([x, bn_1])  # 28, 28, 256
        x = ReLU()(x)

        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            name='deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
        x = BatchNormalization()(x)
        x = keras.layers.add([x, bn_0])  # 56, 56, 256

        '''out_conv_layer'''
        out_heatmap = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='out_heatmap')(x)

        revised_model = Model(inp, [
            out_heatmap,
        ])

        revised_model.summary()
        # plot_model(revised_model, to_file='mnv2_hm.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("mnv2_hm_r_v2.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def mnv2_hm_r_v1(self, tensor, inception_mode=False):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)

        mobilenet_model.layers.pop()

        inp = mobilenet_model.input

        '''block_1 {  block_1_project_BN 56, 56, 24 '''
        x = mobilenet_model.get_layer('block_1_project_BN').output  # 56, 56, 24
        x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', name='b1_conv_1')(x)  # 56, 56, 32
        b1_bn_1 = BatchNormalization(name='b1_bn_1')(x)  # 56, 56, 32
        '''block_1 }'''

        '''-------------------------------------'''

        '''block_3 {  block_3_project_BN 28, 28, 32 '''
        x = mobilenet_model.get_layer('block_3_project_BN').output  # 28, 28, 32
        x = keras.layers.add([x, b1_bn_1])  # 28, 28, 32
        x = ReLU(6.)(x)

        '''up-> 56'''
        x = Deconvolution2D(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                             kernel_initializer='he_uniform')(x)  # 56, 56, 32
        x = BatchNormalization()(x)  # 56, 56, 32
        block_3_heatmap = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='block_3_heatmap')(x)

        '''down-> 14'''
        x = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)  # 28, 28, 64
        b2_bn_28 = BatchNormalization(name='b2_bn_28')(x)  # 28, 28, 64
        x = ReLU(6.)(b2_bn_28)

        x = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)  # 14, 14, 64
        b3_bn_14 = BatchNormalization(name='b3_bn_14')(x)  # 14, 14, 64
        '''block_3 }'''

        '''-------------------------'''

        '''block_6 {  block_6_project_BN 14, 14, 64 '''
        x = mobilenet_model.get_layer('block_6_project_BN').output  # 14, 14, 64
        x = keras.layers.add([x, b3_bn_14])  # 14, 14, 64
        x = ReLU(6.)(x)

        '''up-> 56'''
        x = Deconvolution2D(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            kernel_initializer='he_uniform')(x)  # 28, 28, 32
        x = BatchNormalization()(x)  # 28, 28, 32
        x = keras.layers.add([x, b2_bn_28])  # 28, 28, 32

        x = Deconvolution2D(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            kernel_initializer='he_uniform')(x)  # 56, 56, 32
        x = BatchNormalization()(x)  # 56, 56, 32
        block_6_heatmap = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='block_6_heatmap')(x)


        '''down-> 14'''
        x = Conv2D(96, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)  # 28, 28, 64
        b6_bn_28 = BatchNormalization(name='b6_bn_28')(x)  # 28, 28, 64
        x = ReLU(6.)(b6_bn_28)

        x = Conv2D(96, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)  # 14, 14, 96
        b6_bn_14 = BatchNormalization(name='b6_bn_56')(x)  # 14, 14, 64
        '''block_6 }'''

        '''-------------------------'''

        '''block_10 {  block_10_project_BN 14, 14, 96 '''
        x = mobilenet_model.get_layer('block_10_project_BN').output  # 14, 14, 96
        x = keras.layers.add([x, b6_bn_14])  # 14, 14, 96
        x = ReLU(6.)(x)

        '''up-> 56'''
        x = Deconvolution2D(filters=96, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            kernel_initializer='he_uniform')(x)  # 28, 28, 64
        x = BatchNormalization()(x)  # 28, 28, 64
        x = keras.layers.add([x, b6_bn_28])  # 28, 28, 64

        x = Deconvolution2D(filters=68, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            kernel_initializer='he_uniform')(x)  # 56, 56, 96
        x = BatchNormalization()(x)  # 56, 56, 96
        block_10_heatmap = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='block_10_heatmap')(x)


        '''down-> 7'''
        x = Conv2D(160, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)  # 28, 28, 32
        b10_bn_28 = BatchNormalization(name='b10_bn_28')(x)  # 28, 28, 64
        x = ReLU(6.)(b10_bn_28)

        x = Conv2D(160, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)  # 14, 14, 96
        b10_bn_14 = BatchNormalization(name='b10_bn_14')(x)  # 14, 14, 96
        x = ReLU(6.)(b10_bn_14)

        x = Conv2D(160, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)  # 7, 7, 160
        b10_bn_7 = BatchNormalization(name='b10_bn_7')(x)  # 7, 7, 160
        '''block_10 }'''

        '''-------------------------'''

        '''block_13 {  block_13_project_BN 7, 7, 160'''
        x = mobilenet_model.get_layer('block_13_project_BN').output  # 7, 7, 160
        x = keras.layers.add([x, b10_bn_7])  # 7, 7, 160
        x = ReLU(6.)(x)

        '''up-> 56'''
        x = Deconvolution2D(filters=160, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            kernel_initializer='he_uniform')(x)  # 14, 14, 160
        x = BatchNormalization()(x)  # 14, 14, 160
        x = keras.layers.add([x, b10_bn_14])  # 14, 14, 160

        x = Deconvolution2D(filters=160, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            kernel_initializer='he_uniform')(x)  # 28, 28, 160
        x = BatchNormalization()(x)  # 28, 28, 64
        x = keras.layers.add([x, b10_bn_28])  # 28, 28, 160

        x = Deconvolution2D(filters=68, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            kernel_initializer='he_uniform')(x)  # 56, 56, 160
        x = BatchNormalization()(x)  # 56, 56, 96
        block_13_heatmap = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='block_13_heatmap')(x)

        '''down-> 7'''
        x = Conv2D(160, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)  # 28, 28, 160
        b13_bn_28 = BatchNormalization(name='b13_bn_28')(x)  # 14, 14, 64
        x = ReLU(6.)(b13_bn_28)

        x = Conv2D(160, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)  # 14, 14, 160
        b13_bn_14 = BatchNormalization(name='b13_bn_14')(x)  # 14, 14, 64
        x = ReLU(6.)(b13_bn_14)

        x = Conv2D(320, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)  # 7, 7, 320
        b13_bn_7 = BatchNormalization(name='b13_bn_7')(x)  # 14, 14, 64
        '''block_13 }'''

        '''-------------------------'''

        '''block_16 {  block_16_project_BN 7, 7, 320'''
        x = mobilenet_model.get_layer('block_16_project_BN').output  # 7, 7, 320
        x = keras.layers.add([x, b13_bn_7])  # 7, 7, 320
        x = ReLU(6.)(x)

        '''up-> 56'''
        x = Deconvolution2D(filters=160, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            kernel_initializer='he_uniform')(x)  # 14, 14, 160
        x = BatchNormalization()(x)  # 14, 14, 160
        x = keras.layers.add([x, b13_bn_14])  # 14, 14, 160

        x = Deconvolution2D(filters=160, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            kernel_initializer='he_uniform')(x)  # 28, 28, 160
        x = BatchNormalization()(x)  # 28, 28, 64
        x = keras.layers.add([x, b13_bn_28])  # 28, 28, 160

        x = Deconvolution2D(filters=68, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            kernel_initializer='he_uniform')(x)  # 56, 56, 160
        x = BatchNormalization()(x)  # 56, 56, 96
        block_16_heatmap = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='block_16_heatmap')(x)

        '''down-> 7'''
        x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)  # 28, 28, 32
        b16_bn_28 = BatchNormalization(name='b16_bn_28')(x)  # 14, 14, 64
        x = ReLU(6.)(b10_bn_28)

        x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)  # 14, 14, 32
        b16_bn_14 = BatchNormalization(name='b16_bn_14')(x)  # 14, 14, 64
        x = ReLU(6.)(b10_bn_14)

        x = Conv2D(1280, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)  # 7, 7, 32
        b16_bn_7 = BatchNormalization(name='b16_bn_7')(x)  # 7, 7, 32
        '''block_16 }'''

        '''-------------------------'''
        '''block_out {  out_relu 7, 7, 1280'''
        x = mobilenet_model.get_layer('out_relu').output  # 7, 7, 1280
        x = keras.layers.add([x, b16_bn_7])  # 7, 7, 1280
        x = ReLU(6.)(x)

        '''up-> 56'''
        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            kernel_initializer='he_uniform')(x)  # 14, 14, 160
        x = BatchNormalization()(x)  # 14, 14, 160
        x = keras.layers.add([x, b16_bn_14])  # 14, 14, 160

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            kernel_initializer='he_uniform')(x)  # 28, 28, 160
        x = BatchNormalization()(x)  # 28, 28, 64
        x = keras.layers.add([x, b16_bn_28])  # 28, 28, 160

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            kernel_initializer='he_uniform')(x)  # 56, 56, 160
        x = BatchNormalization()(x)  # 56, 56, 96

        out_heatmap = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='out_heatmap')(x)

        '''block_out }'''
        if inception_mode:
            revised_model = Model(inp, [
                out_heatmap,
                block_3_heatmap,
                block_6_heatmap,
                block_10_heatmap,
                block_13_heatmap,
                block_16_heatmap
            ])
        else:
            revised_model = Model(inp, [
                out_heatmap
            ])

        revised_model.summary()
        # plot_model(revised_model, to_file='mnv2_hm_r.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("mnv2_hm_r.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def mnv2_hm_r_v0(self, tensor):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)

        mobilenet_model.layers.pop()

        inp = mobilenet_model.input


        '''block_1 {  block_3_project_BN 28, 28, 32 '''
        x = mobilenet_model.get_layer('block_3_project_BN').output  # 28, 28, 32
        # x = ReLU(6.)(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='block_1_deconv1', kernel_initializer='he_uniform')(x)  # 56, 56, 128
        block_1__bn1 = BatchNormalization(name='block_1__bn1')(x)

        '''block_1 }'''


        '''block_2 {  block_10_project_BN 14, 14, 96 '''
        x = mobilenet_model.get_layer('block_10_project_BN').output  # 14, 14, 96
        # x = ReLU(6.)(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='block_2_deconv1', kernel_initializer='he_uniform')(x)  # 28, 28, 128
        block_2__bn1 = BatchNormalization(name='block_2__bn1')(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='block_2_deconv2', kernel_initializer='he_uniform')(block_2__bn1)  # 56, 56, 128
        block_2__bn2 = BatchNormalization(name='block_2_bn2')(x)
        block_2__bn2 = keras.layers.add([block_2__bn2, block_1__bn1], name='block_2_add_1')

        block_2__bn2 = Conv2D(128, kernel_size=1, padding='same', name='block_2_conv_1')(block_2__bn2)
        block_2__bn2 = BatchNormalization(name='block_2_bn3')(block_2__bn2)
        block_2__bn2 = ReLU(6.)(block_2__bn2)
        '''block_2 }'''


        '''block_3 {  block_13_project_BN 7, 7, 160 '''
        x = mobilenet_model.get_layer('block_13_project_BN').output  # 7, 7, 160
        # x = ReLU(6.)(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='block_3_deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 128
        block_3__bn1 = BatchNormalization(name='block_3__bn1')(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='block_3_deconv2', kernel_initializer='he_uniform')(block_3__bn1)  # 28, 28, 128
        block_3__bn2 = BatchNormalization(name='block_3__bn2')(x)
        block_3__bn2 = keras.layers.add([block_3__bn2, block_2__bn1], name='block_3_add_1')

        x = Conv2D(128, kernel_size=1, padding='same', name='block_3_conv_2')(block_3__bn2)
        x = BatchNormalization(name='block_3_bn3_1')(x)
        block_3_bn3_1 = ReLU(6.)(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='block_3_deconv3', kernel_initializer='he_uniform')(block_3__bn2)  # 56, 56, 128
        block_3__bn3 = BatchNormalization(name='block_3__bn3')(x)
        block_3__bn3 = keras.layers.add([block_3__bn3, block_2__bn2], name='block_3_add_2')

        block_3__bn3 = Conv2D(128, kernel_size=1, padding='same', name='block_3_conv_1')(block_3__bn3)
        block_3__bn3 = BatchNormalization(name='block_3_bn4')(block_3__bn3)
        block_3__bn3 = ReLU(6.)(block_3__bn3)
        '''block_3 }'''

        '''out{'''
        x = mobilenet_model.get_layer('block_16_project_BN').output  # 7, 7, 320
        x = ReLU(6.)(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='out_deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
        out_bn1 = BatchNormalization(name='out_bn1')(x)
        x = keras.layers.add([out_bn1, block_3__bn1], name='out_add_1')

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='out_deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
        out_bn2 = BatchNormalization(name='out_bn2')(x)
        x = keras.layers.add([out_bn2, block_3_bn3_1], name='out_add_2')
        # x = keras.layers.add([out_bn2, block_3__bn2], name='out_add_2')

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='out_deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
        out_bn3 = BatchNormalization(name='out_bn3')(x)
        x = keras.layers.add([out_bn3, block_3__bn3], name='out_add_3')

        '''out}'''

        out_heatmap = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='out_heatmap')(x)

        revised_model = Model(inp, [
            out_heatmap,
        ])

        revised_model.summary()
        plot_model(revised_model, to_file='mnv2_hm_r.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("mnv2_hm_r.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def mnv2_hm(self, tensor):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)

        mobilenet_model.layers.pop()

        inp = mobilenet_model.input
        '''heatmap can not be generated from activation layers, so we use out_relu'''
        x = mobilenet_model.get_layer('out_relu').output  # 7, 7, 1280
        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
        x = BatchNormalization(name='out_bn1')(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
        x = BatchNormalization(name='out_bn2')(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
        x = BatchNormalization(name='out_bn3')(x)

        out_heatmap = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='out_heatmap')(x)

        revised_model = Model(inp, [
            out_heatmap,
        ])

        revised_model.summary()
        # plot_model(revised_model, to_file='mnv2_hm.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("mnv2_hm.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def resNet_50_main_inception_both_v1(self, tensor):
        resnet_model = resnet50.ResNet50(include_top=True,
                                         weights=None,
                                         input_tensor=tensor,
                                         input_shape=[224, 224, 3],
                                         pooling=None)

        resnet_model.layers.pop()

        inp = resnet_model.input

        '''inception_1      activation_13       28, 28, 512  '''
        '''heatmap {'''
        x = resnet_model.get_layer('activation_13').output  # 28, 28, 512
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='incp_1_deconv1', kernel_initializer='he_uniform')(x)  # 56, 56, 256
        incp_1_bn1 = BatchNormalization(name='incp_1_bn1')(x)
        '''} heatmap'''
        '''} inception_1 '''

        '''inception_2      activation_25      14, 14, 1024   '''
        '''heatmap{ '''
        x = resnet_model.get_layer('activation_25').output  # 14, 14, 1024
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='incp_2_deconv1', kernel_initializer='he_uniform')(x)  # 28, 28, 256
        incp_2_bn1 = BatchNormalization(name='incp_2_bn1')(x)
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='incp_2_deconv2', kernel_initializer='he_uniform')(incp_2_bn1)  # 56, 56, 256
        incp_2_bn2 = BatchNormalization(name='incp_2_bn2')(x)
        incp_2_add_1 = keras.layers.add([incp_1_bn1, incp_2_bn2], name='incp_2_add_1')
        '''} heatmap'''
        '''} inception_2 '''

        '''inception _3      activation_43      7, 7, 2048  {'''
        '''heatmap{  '''
        x = resnet_model.get_layer('activation_43').output  # 7, 7, 2048
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='incp_3_deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
        incp_3_bn1 = BatchNormalization(name='incp_3_bn1')(x)

        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='incp_3_deconv2', kernel_initializer='he_uniform')(incp_3_bn1)  # 28, 28, 256
        incp_3_bn2 = BatchNormalization(name='incp_3_bn2')(x)
        incp_3_add_1 = keras.layers.add([incp_3_bn2, incp_2_bn1], name='incp_3_add_1')

        x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='incp_3_deconv3', kernel_initializer='he_uniform')(incp_3_add_1)  # 56, 56, 256
        incp_3_bn3 = BatchNormalization(name='incp_3_bn3')(x)
        incp_3_add_2 = keras.layers.add([incp_3_bn3, incp_2_add_1], name='incp_3_add_2')

        '''} heatmap'''

        '''} inception _3 '''

        '''output {'''
        '''heatmap {'''
        x = resnet_model.get_layer('activation_49').output  # 7, 7, 2048
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
        x = BatchNormalization(name='out_bn1')(x)
        x = keras.layers.add([x, incp_3_bn1], )

        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
        x = BatchNormalization(name='out_bn2')(x)
        x = keras.layers.add([x, incp_3_add_1])

        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
        x = BatchNormalization(name='out_bn3')(x)
        x = keras.layers.add([x, incp_3_add_2])

        out_heatmap = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='out_heatmap')(x)
        '''} heatmap '''
        '''} output '''

        revised_model = Model(inp, [
            out_heatmap,
        ])

        revised_model.summary()
        # plot_model(revised_model, to_file='resnet_50_incep_BOTH.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("resnet_50_incep_BOTH.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def resNet_50_main_inception_both_v0(self, tensor):
        resnet_model = resnet50.ResNet50(include_top=True,
                                         weights=None,
                                         input_tensor=tensor,
                                         input_shape=[224, 224, 3],
                                         pooling=None)

        resnet_model.layers.pop()

        inp = resnet_model.input

        '''inception_1      activation_13       28, 28, 512  '''
        '''heatmap {'''
        x = resnet_model.get_layer('activation_13').output  # 28, 28, 512
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='incp_1_deconv1', kernel_initializer='he_uniform')(x)  # 56, 56, 256
        incp_1_bn1 = BatchNormalization(name='incp_1_bn1')(x)
        out_inc1_heatmap = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='out_inc1_heatmap')(incp_1_bn1)
        out_inc1_heatmap_all = Conv2D(1, kernel_size=1, padding='same', name='out_inc1_heatmap_all')(incp_1_bn1)
        '''} heatmap'''
        '''regression branch to create pose and points {'''
        # x = resnet_model.get_layer('activation_13').output  # 2048
        # x = GlobalAveragePooling2D()(x)
        # x = Dense(LearningConfig.landmark_len + InputDataSize.pose_len, name='dense_1_1', activation='relu',
        #           kernel_initializer='he_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
        # x = Dense(LearningConfig.landmark_len + InputDataSize.pose_len, name='dense_1_2', activation='relu',
        #           kernel_initializer='he_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
        #
        # out_inc1_face = Dense(LearningConfig.landmark_len, name='incp_1_out_face')(x)
        # out_inc1_pose = Dense(InputDataSize.pose_len, name='incp_1_out_pose')(x)
        '''regression branch }'''
        '''} inception_1 '''

        '''inception_2      activation_25      14, 14, 1024   '''
        '''heatmap{ '''
        x = resnet_model.get_layer('activation_25').output  # 14, 14, 1024
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='incp_2_deconv1', kernel_initializer='he_uniform')(x)  # 28, 28, 256
        incp_2_bn1 = BatchNormalization(name='incp_2_bn1')(x)
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='incp_2_deconv2', kernel_initializer='he_uniform')(incp_2_bn1)  # 56, 56, 256
        incp_2_bn2 = BatchNormalization(name='incp_2_bn2')(x)
        incp_2_add_1 = keras.layers.add([incp_1_bn1, incp_2_bn2], name='incp_2_add_1')
        out_inc2_heatmap = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='out_inc2_heatmap')(incp_2_add_1)
        out_inc2_heatmap_all = Conv2D(1, kernel_size=1, padding='same', name='out_inc2_heatmap_all')(incp_2_add_1)
        '''} heatmap'''
        '''regression branch to create pose and points {'''
        # x = resnet_model.get_layer('activation_25').output  # 2048
        # x = GlobalAveragePooling2D()(x)
        # x = Dense(LearningConfig.landmark_len + InputDataSize.pose_len, name='dense_2_1', activation='relu',
        #           kernel_initializer='he_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
        # x = Dense(LearningConfig.landmark_len + InputDataSize.pose_len, name='dense_2_2', activation='relu',
        #           kernel_initializer='he_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
        #
        # out_inc2_face = Dense(LearningConfig.landmark_len, name='incp_2_out_face')(x)
        # out_inc2_pose = Dense(InputDataSize.pose_len, name='incp_2_out_pose')(x)
        '''regression branch }'''
        '''} inception_2 '''

        '''inception _3      activation_43      7, 7, 2048  {'''
        '''heatmap{  '''
        x = resnet_model.get_layer('activation_43').output  # 7, 7, 2048
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='incp_3_deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
        incp_3_bn1 = BatchNormalization(name='incp_3_bn1')(x)

        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='incp_3_deconv2', kernel_initializer='he_uniform')(incp_3_bn1)  # 28, 28, 256
        incp_3_bn2 = BatchNormalization(name='incp_3_bn2')(x)
        incp_3_add_1 = keras.layers.add([incp_3_bn2, incp_2_bn1], name='incp_3_add_1')

        x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='incp_3_deconv3', kernel_initializer='he_uniform')(incp_3_add_1)  # 56, 56, 256
        incp_3_bn3 = BatchNormalization(name='incp_3_bn3')(x)
        incp_3_add_2 = keras.layers.add([incp_3_bn3, incp_2_add_1], name='incp_3_add_2')

        out_inc3_heatmap = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='out_inc3_heatmap')(incp_3_add_2)
        out_inc3_heatmap_all = Conv2D(1, kernel_size=1, padding='same', name='out_inc3_heatmap_all')(incp_3_add_2)
        '''} heatmap'''
        '''regression branch to create pose and points {'''
        # x = resnet_model.get_layer('activation_43').output  # 2048
        # x = GlobalAveragePooling2D()(x)
        # x = Dense(LearningConfig.landmark_len + InputDataSize.pose_len, name='dense_3_1', activation='relu',
        #           kernel_initializer='he_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
        # x = Dense(LearningConfig.landmark_len + InputDataSize.pose_len, name='dense_3_2', activation='relu',
        #           kernel_initializer='he_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
        #
        # out_inc3_face = Dense(LearningConfig.landmark_len, name='incp_3_out_face')(x)
        # out_inc3_pose = Dense(InputDataSize.pose_len, name='incp_3_out_pose')(x)
        '''regression branch }'''
        '''} inception _3 '''

        '''output {'''
        '''heatmap {'''
        x = resnet_model.get_layer('activation_49').output  # 7, 7, 2048
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
        x = BatchNormalization(name='out_bn1')(x)
        x = keras.layers.add([x, incp_3_bn1],)

        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
        x = BatchNormalization(name='out_bn2')(x)
        x = keras.layers.add([x, incp_3_add_1])

        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
        x = BatchNormalization(name='out_bn3')(x)
        x = keras.layers.add([x, incp_3_add_2])

        out_heatmap = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='out_heatmap')(x)
        out_heatmap_all = Conv2D(1, kernel_size=1, padding='same', name='out_heatmap_all')(x)
        '''} heatmap '''

        '''regression branch to create pose and points {'''
        x = resnet_model.get_layer('avg_pool').output  # 2048

        x = Dense(LearningConfig.landmark_len + InputDataSize.pose_len, name='dense_o_1', activation='relu',
                  kernel_initializer='he_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
        x = Dense(LearningConfig.landmark_len + InputDataSize.pose_len, name='dense_o_2', activation='relu',
                  kernel_initializer='he_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)

        out_face = Dense(LearningConfig.landmark_len, name='out_face')(x)
        out_pose = Dense(InputDataSize.pose_len, name='out_pose')(x)
        '''regression branch }'''
        '''} output '''


        # revised_model = Model(inp, [
        #                             out_heatmap, out_heatmap_all, out_face, out_pose,
        #                             out_inc1_heatmap, out_inc1_heatmap_all, out_inc1_face, out_inc1_pose,
        #                             out_inc2_heatmap, out_inc2_heatmap_all, out_inc2_face, out_inc2_pose,
        #                             out_inc3_heatmap, out_inc3_heatmap_all, out_inc3_face, out_inc3_pose
        #                             ])
        revised_model = Model(inp, [
                                    out_heatmap, out_heatmap_all, out_face, out_pose,
                                    out_inc1_heatmap, out_inc1_heatmap_all,
                                    out_inc2_heatmap, out_inc2_heatmap_all,
                                    out_inc3_heatmap, out_inc3_heatmap_all
                                    ])

        revised_model.summary()
        # plot_model(revised_model, to_file='resnet_50_incep_BOTH.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("resnet_50_incep_BOTH.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def resNet_50_main_both(self, tensor):
        resnet_model = resnet50.ResNet50(include_top=True,
                                         weights=None,
                                         input_tensor=tensor,
                                         input_shape=[224, 224, 3],
                                         pooling=None)

        resnet_model.layers.pop()

        inp = resnet_model.input

        '''deconv branch to create heatmaps {'''
        x = resnet_model.get_layer('activation_49').output  # 2048
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='deconv1', kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='deconv2', kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='deconv3', kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        out_deconv = Conv2D(LearningConfig.landmark_len//2, kernel_size=1, padding='same', name='conv_out')(x)
        '''deconv branch}'''

        '''regression branch to create pose and points {'''
        x = resnet_model.get_layer('avg_pool').output  # 2048

        x = Dense(LearningConfig.landmark_len + InputDataSize.pose_len, name='dense_1', activation='relu',
                  kernel_initializer='he_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
        x = Dense(LearningConfig.landmark_len + InputDataSize.pose_len, name='dense_2', activation='relu',
                  kernel_initializer='he_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)

        out_face = Dense(LearningConfig.landmark_len, name='out_face')(x)
        out_pose = Dense(InputDataSize.pose_len, name='out_pose')(x)
        '''regression branch }'''


        revised_model = Model(inp, [out_deconv, out_face, out_pose])

        revised_model.summary()
        model_json = revised_model.to_json()

        with open("resnet_50_revised_reg_deconv.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def resNet_50_main_inception_deconv(self,tensor):
        resnet_model = resnet50.ResNet50(include_top=True,
                                         weights=None,
                                         input_tensor=tensor,
                                         input_shape=[224, 224, 3],
                                         pooling=None)

        resnet_model.layers.pop()

        inp = resnet_model.input

        '''inception _1      activation_13       28, 28, 512  '''
        x = resnet_model.get_layer('activation_13').output  # 28, 28, 512
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='incp_1_deconv1', kernel_initializer='he_uniform')(x)  # 56, 56, 256
        x = BatchNormalization()(x)
        x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu',
                            name='incp_1_deconv2', kernel_initializer='he_uniform')(x)  # 56, 56, 256
        x = BatchNormalization()(x)
        x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu',
                            name='incp_1_deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
        x = BatchNormalization()(x)
        out_inc1 = Conv2D(LearningConfig.landmark_len//2, kernel_size=1, padding='same', name='incp_1_conv_out')(x)

        '''inception _2      activation_25      14, 14, 1024   '''
        x = resnet_model.get_layer('activation_25').output  # 14, 14, 1024
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='incp_2_deconv1', kernel_initializer='he_uniform')(x)  # 28, 28, 256
        x = BatchNormalization()(x)
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='incp_2_deconv2', kernel_initializer='he_uniform')(x)  # 56, 56, 256
        x = BatchNormalization()(x)
        x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu',
                            name='incp_2_deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
        x = BatchNormalization()(x)
        out_inc2 = Conv2D(LearningConfig.landmark_len//2, kernel_size=1, padding='same', name='incp_2_conv_out')(x)

        '''inception _3      activation_43      7, 7, 2048  '''
        x = resnet_model.get_layer('activation_43').output  # 7, 7, 2048
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='incp_3_deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
        x = BatchNormalization()(x)
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='incp_3_deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
        x = BatchNormalization()(x)
        x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='incp_3_deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
        x = BatchNormalization()(x)
        out_inc3 = Conv2D(LearningConfig.landmark_len//2, kernel_size=1, padding='same', name='incp_3_conv_out')(x)


        '''output '''
        x = resnet_model.get_layer('activation_49').output  # 7, 7, 2048
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='deconv1', kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='deconv2', kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                            name='deconv3', kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        out = Conv2D(LearningConfig.landmark_len//2, kernel_size=1, padding='same', name='conv_out')(x)

        revised_model = Model(inp, [out, out_inc1, out_inc2, out_inc3])

        revised_model.summary()
        model_json = revised_model.to_json()

        with open("resnet_50_incep_deconv.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def resNet_50_main_deconv(self,tensor):
        resnet_model = resnet50.ResNet50(include_top=True,
                                         weights=None,
                                         input_tensor=tensor,
                                         input_shape=[224, 224, 3],
                                         pooling=None)

        resnet_model.layers.pop()

        inp = resnet_model.input

        x = resnet_model.get_layer('activation_49').output  # 2048
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu', name='deconv1', kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu', name='deconv2', kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Deconvolution2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu', name='deconv3', kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        out = Conv2D(LearningConfig.landmark_len, kernel_size=1, padding='same', name='conv_out')(x)

        revised_model = Model(inp, out)

        revised_model.summary()
        model_json = revised_model.to_json()

        with open("resnet_50_revised_deconv.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def denseNet_main(self, tensor, pose, multitask):
        densenet_model = densenet.DenseNet121(include_top=True,
                                                weights=None,
                                                input_tensor=None,
                                                input_shape=None,
                                                pooling=None)
        densenet_model.summary()
        model_json = densenet_model.to_json()

        with open("000denseNet201.json", "w") as json_file:
            json_file.write(model_json)

    def mobileNet_v2_main(self, tensor):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)
        # , classes=cnf.landmark_len)

        mobilenet_model.layers.pop()

        x = mobilenet_model.get_layer('global_average_pooling2d_1').output  # 1280
        x = Dense(LearningConfig.landmark_len, name='dense_layer_out_2', activation='relu',
                  kernel_initializer='he_uniform')(x)
        out = Dense(LearningConfig.landmark_len, name='out')(x)

        inp = mobilenet_model.input

        revised_model = Model(inp, out)

        revised_model.summary()
        # plot_model(revised_model, to_file='mobileNet_v2_main.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("mobileNet_v2_main.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def __print_figures(self, history):
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        mean_squared_error = history.history['mean_squared_error']
        val_mean_squared_error = history.history['val_mean_squared_error']

        mean_absolute_error = history.history['mean_absolute_error']
        val_mean_absolute_error = history.history['val_mean_absolute_error']

        mean_absolute_percentage_error = history.history['mean_absolute_percentage_error']
        val_mean_absolute_percentage_error = history.history['val_mean_absolute_percentage_error']

        epochs = range(1, len(loss) + 1)

        # 3 loss imgs
        plt.figure()
        plt.title('Training and validation loss')
        plt.plot(epochs, loss, 'red', label='Training loss')
        plt.plot(epochs, val_loss, 'blue', label='Validation loss')
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig('loss.png')

        # mes
        plt.figure()
        plt.title('Training and validation mse')
        plt.plot(epochs, mean_squared_error, 'red', label='mse')
        plt.plot(epochs, val_mean_squared_error, 'blue', label='val_mse')
        plt.xlabel("epoch")
        plt.ylabel("mse")
        plt.legend()
        plt.savefig('mse_1.png')

        # mae
        plt.figure()
        plt.title('Training and validation mae')
        plt.plot(epochs, mean_absolute_error, label='mae')
        plt.plot(epochs, val_mean_absolute_error, label='val_mae')
        plt.xlabel("epoch")
        plt.ylabel("mae")
        plt.legend()
        plt.savefig('mae_1.png')

        # mape img
        plt.figure()
        plt.title('Training and validation mape')
        plt.plot(epochs, mean_absolute_percentage_error, label='mape')
        plt.plot(epochs, val_mean_absolute_percentage_error, label='val_mape')
        plt.xlabel("epoch")
        plt.ylabel("mape")
        plt.legend()
        plt.savefig('mape_1.png')

    def _post_process_correction(self, predicted_landmarks, normalized=False):
        if normalized:
            width = 224
            height = 224
            x_center = width / 2
            y_center = height / 2
            landmark_arr_flat_normalized = []
            for p in range(0, len(predicted_landmarks), 2):
                landmark_arr_flat_normalized.append((x_center - predicted_landmarks[p]) / width)
                landmark_arr_flat_normalized.append((y_center - predicted_landmarks[p + 1]) / height)
            predicted_landmarks = landmark_arr_flat_normalized

        predicted_landmarks = np.array(predicted_landmarks)
        predicted_landmarks = predicted_landmarks.reshape(-1)
        pca_utility = PCAUtility()
        eigenvalues, eigenvectors, meanvector = pca_utility.load_pca_obj(DatasetName.ibug)

        b_vector_p = self.calculate_b_vector(predicted_landmarks, True, eigenvalues, eigenvectors, meanvector)
        y_pre_asm = meanvector + np.dot(eigenvectors, b_vector_p)
        # print('-----------------')
        # print(predicted_landmarks)
        # print('--')
        # print(y_pre_asm)
        # print('-------------------')

        return y_pre_asm
