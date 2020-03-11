from skimage.io import imread
from skimage.transform import resize
import numpy as np
import tensorflow as tf
import keras
from skimage.transform import resize
from tf_record_utility import  TFRecordUtility
from configuration import DatasetName, DatasetType, \
    AffectnetConf, IbugConf, W300Conf, InputDataSize, LearningConfig
from numpy import save, load, asarray
from cnn_model import CNNModel
import img_printer as imgpr
from keras import backend as K

class DataPWGenerator(keras.utils.Sequence):

    def __init__(self, image_filenames, label_filenames, batch_size, n_outputs):
        self.image_filenames = image_filenames
        self.label_filenames = label_filenames
        self.batch_size = batch_size
        self.n_outputs = n_outputs

    def __len__(self):
        _len = np.ceil(len(self.image_filenames) // float(self.batch_size))
        return int(_len)

    def __getitem__(self, idx):

        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.label_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        img_batch = np.array([self._image_read(file_name) for file_name in batch_x])
        lbl_batch = np.array([self._lbl_prepration(file_name) for file_name in batch_y])

        lbl_out_array = []
        img_out_array = []
        for i in range(self.n_outputs):
            lbl_out_array.append(lbl_batch)
            img_out_array.append(img_batch)

        return img_out_array, lbl_out_array

    def _lbl_prepration(self, file_name):
        return np.zeros(shape=[2])

        lbl = load(file_name)
        return lbl

    def _image_read(self, file_name):
        return np.zeros(shape=[32, 32, 3])

        input = load(file_name)
        return input