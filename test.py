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
from image_utility import ImageUtility
import img_printer as imgpr
from PIL import Image


class Test:

    def __init__(self):
        self.test()

    def test(self):
        image_utility = ImageUtility()
        cnn = CNNModel()

        images_dir_out_ch = '/media/ali/extradata/facial_landmark_ds/from_ibug/test_set/challenging_intermediate_img/'

        model = cnn.mobileNet_v2_small(tensor=None)
        model.load_weights('weights-03-0.00095.h5')

        counter =0

        for img_file in os.listdir(images_dir_out_ch):
            if img_file.endswith("_img.npy"):
                lbl_file = img_file[:-8] + "_lbl.npy"
                main_img_file = img_file[:-8] + ".jpg"

                img = Image.open(images_dir_out_ch + main_img_file)

                image = np.expand_dims(load(images_dir_out_ch + img_file), axis=0)
                lbl = load(images_dir_out_ch + lbl_file)

                p_lbl = model.predict(image)[0]

                labels_predict_transformed, landmark_arr_x_p, landmark_arr_y_p = image_utility.\
                    create_landmarks_from_normalized(p_lbl, 224, 224, 112, 112)

                imgpr.print_image_arr((counter+1)*1000, img, landmark_arr_x_p, landmark_arr_y_p)
                counter += 1




