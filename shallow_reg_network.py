from configuration import DatasetName, DatasetType, \
    AffectnetConf, IbugConf, W300Conf, InputDataSize, LearningConfig
from hg_Class import HourglassNet

import tensorflow as tf
import keras
from skimage.transform import resize

from keras.regularizers import l2
tf.logging.set_verbosity(tf.logging.ERROR)
from keras.models import Model
from keras.applications import mobilenet_v2, mobilenet, resnet50, densenet
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, \
    BatchNormalization, Activation, GlobalAveragePooling2D, DepthwiseConv2D, \
    Dropout, ReLU, Concatenate, Deconvolution2D, Input, add

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


class ShallowRegNetwork(object):
    def __init__(self, inp_shape, out_shape, num_branches, num_filters):

        self.inp_shape = inp_shape
        self.out_shape = out_shape
        self.num_branches = num_branches
        self.num_filters = num_filters

    def create_model(self):
        branches = []
        inputs = []
        for i in range(self.num_branches):
            inp_i, br_i = self.create_branch(str(i))
            inputs.append(inp_i)
            branches.append(br_i)

        # concat_1 = Concatenate()(branches)
        # out_1 = ReLU(6., name='out_1_relu')(concat_1)
        # out_1 = GlobalAveragePooling2D(name='out_1_GlobalAveragePooling2D')(out_1)
        #
        # out_s = [Dense(out_shape, name=str(i)+'_out')(out_1) for i in range(num_branches)]

        # revised_model = Model(inputs, out_s)
        revised_model = Model(inputs, branches)

        revised_model.summary()

        model_json = revised_model.to_json()
        with open("ShallowRegNetwork.json", "w") as json_file:
            json_file.write(model_json)
        return revised_model

    def create_branch(self, branch_prefix):
        """inp_shape """
        _depth_multiplier = 2
        input_1 = Input(shape=(self.inp_shape, self.inp_shape, 1))

        b1_cnv2d = Conv2D(filters=self.num_filters, kernel_size=(3, 3), strides=(2, 2), padding='same',
                          use_bias=False, name=branch_prefix+'_b1_cnv2d', kernel_initializer='normal')(input_1)
        b1_relu = ReLU(6., name=branch_prefix+'_b1_relu')(b1_cnv2d)
        b1_bn = BatchNormalization(epsilon=1e-3, momentum=0.999, name=branch_prefix+'_b1_bn')(b1_relu)

        '''b2'''
        b2_dw2d = DepthwiseConv2D(kernel_size=3, strides=1, use_bias=False,
                                  padding='same', name=branch_prefix+'_b2_dw2d')(b1_bn)
        b2_relu = ReLU(6., name=branch_prefix+'_b2_relu')(b2_dw2d)
        b2_bn = BatchNormalization(epsilon=1e-3, momentum=0.999, name=branch_prefix+'_b2_bn')(b2_relu)
        add_1 = add([b1_bn, b2_bn])  # 16 * 16

        '''b3_new_branch'''
        b3_relu = ReLU(6., name=branch_prefix+'b3_relu')(add_1)
        b3_cnv2d = Conv2D(filters=self.num_filters*2, kernel_size=(3, 3), strides=(2, 2), padding='same',
                          use_bias=False, name=branch_prefix+'_b3_cnv2d', kernel_initializer='normal')(b3_relu)
        b3_relu_2 = ReLU(6., name=branch_prefix+'_b3_relu_2')(b3_cnv2d)
        b3_bn = BatchNormalization(epsilon=1e-3, momentum=0.999, name=branch_prefix+'_b3_bn')(b3_relu_2)

        '''b4'''
        b4_dw2d = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False,
                                  padding='same', name=branch_prefix+'_b4_dw2d')(b3_bn)
        b4_relu = ReLU(6., name=branch_prefix+'_b4_relu')(b4_dw2d)
        b4_bn = BatchNormalization(epsilon=1e-3, momentum=0.999, name=branch_prefix+'_b4_bn')(b4_relu)
        add_2 = add([b3_bn, b4_bn])   # 8 * 8

        '''b5_new_branch'''
        b5_relu = ReLU(6., name=branch_prefix+'_b5_relu')(add_2)
        b5_cnv2d = Conv2D(filters=self.num_filters*4, kernel_size=(3, 3), strides=(2, 2), padding='same',
                          use_bias=False, name=branch_prefix+'_b5_cnv2d', kernel_initializer='normal')(b5_relu)
        b5_relu_2 = ReLU(6., name=branch_prefix+'_b5_relu_2')(b5_cnv2d)
        b5_bn = BatchNormalization(epsilon=1e-3, momentum=0.999, name=branch_prefix+'b5_bn')(b5_relu_2)

        '''b6'''
        b6_dw2d = DepthwiseConv2D(kernel_size=3, strides=1, activation=None, use_bias=False,
                                  padding='same', name=branch_prefix+'_b6_dw2d')(b5_bn)
        b6_relu = ReLU(6., name=branch_prefix+'_b6_relu')(b6_dw2d)
        b6_bn = BatchNormalization(epsilon=1e-3, momentum=0.999, name=branch_prefix+'_b6_bn')(b6_relu)
        add_3 = add([b5_bn, b6_bn])  # 4 * 4

        '''b7_new_branch'''
        b7_relu = ReLU(6., name=branch_prefix+'b7_relu')(add_3)
        b7_cnv2d = Conv2D(filters=self.num_filters*4, kernel_size=(3, 3), strides=(2, 2), padding='same',
                          use_bias=False, name=branch_prefix+'_b7_cnv2d', kernel_initializer='normal')(b7_relu)
        b7_relu_2 = ReLU(6., name=branch_prefix+'_b7_relu_2')(b7_cnv2d)
        b7_bn = BatchNormalization(epsilon=1e-3, momentum=0.999, name=branch_prefix+'_b7_bn')(b7_relu_2)

        '''b8'''
        b8_dw2d = DepthwiseConv2D(depth_multiplier=1, kernel_size=3, strides=1, activation=None, use_bias=False,
                                  padding='same', name=branch_prefix+'_b8_dw2d')(b7_bn)
        b8_relu = ReLU(6., name=branch_prefix+'_b8_relu')(b8_dw2d)
        b8_bn = BatchNormalization(epsilon=1e-3, momentum=0.999, name=branch_prefix+'_b8_bn')(b8_relu)

        branch = add([b7_bn, b8_bn])

        branch = GlobalAveragePooling2D(name=branch_prefix + '_GlobalAveragePooling2D')(branch)
        branch = Dense(12, name=branch_prefix + '_dense', kernel_regularizer=l2(0.01))(branch)
        branch = Dense(2, name=branch_prefix + '_out')(branch)

        return input_1, branch












