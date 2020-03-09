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

class DataHeatmapGenerator(keras.utils.Sequence):

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

        img_batch = np.array([imread(file_name) for file_name in batch_x])
        lbl_batch = np.array([load(file_name) for file_name in batch_y])

        lbl_out_array = []
        for i in range(self.n_outputs):
            lbl_out_array.append(lbl_batch)

        return img_batch, lbl_out_array
