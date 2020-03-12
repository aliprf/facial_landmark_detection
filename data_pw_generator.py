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
        # batch_y = self.label_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        # both batch_x and batch_y are the same here:
        batch_x_files = []
        batch_y_files = []
        for item in batch_x:
            img_files, lbl_files = self.create_file_names(item)
            batch_x_files.append(img_files)
            batch_y_files.append(lbl_files)

        batch_x_files = np.array(batch_x_files).transpose()
        batch_y_files = np.array(batch_y_files).transpose()

        # print('batch_x_files')
        # print(batch_x_files.shape)
        # print('batch_y_files')
        # print(batch_y_files.shape)

        img_batch = []
        lbl_batch = []

        for i in range(68):

            img_batch.append(np.array([self._image_read(file_name) for file_name in batch_x_files[i]]))
            lbl_batch.append(np.array([self._lbl_prepration(file_name) for file_name in batch_y_files[i]]))

        print('img_batch')
        print(np.array(img_batch).shape)
        print('lbl_batch')
        print(np.array(lbl_batch).shape)

        return img_batch, lbl_batch


    def create_file_names(self, file_root):
        images_dir = IbugConf.train_intermediate_img_dir
        lbls_dir = IbugConf.train_intermediate_lbl_dir

        imgs = []
        lbls = []
        for i in range(68):
            imgs.append(images_dir + str(i + 1) + '/' + file_root + '_' + str(i + 1) + ".npy")
            lbls.append(lbls_dir + str(i + 1) + '/' + file_root + '_' + str(i + 1) + ".npy")
        return imgs, lbls



            # def __getitem__(self, idx):
    #
    #     batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
    #     batch_y = self.label_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
    #
    #     img_batch = np.array([self._image_read(file_name) for file_name in batch_x])
    #     lbl_batch = np.array([self._lbl_prepration(file_name) for file_name in batch_y])
    #
    #     lbl_out_array = []
    #     img_out_array = []
    #     for i in range(self.n_outputs):
    #         lbl_out_array.append(lbl_batch)
    #         img_out_array.append(img_batch)
    #
    #     return img_out_array, lbl_out_array

    def _lbl_prepration(self, file_name):
        # return np.zeros(shape=[2])
        lbl = load(file_name)
        return lbl

    def _image_read(self, file_name):
        # return np.zeros(shape=[32, 32, 1])

        _input = load(file_name)  # 32*32
        _input = np.expand_dims(_input, axis=2)  # 32*32*1
        print(_input.shape)
        return _input