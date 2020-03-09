from tf_record_utility import TFRecordUtility
from configuration import DatasetName, DatasetType, AffectnetConf, IbugConf, W300Conf, InputDataSize
from cnn_model import CNNModel
from pca_utility import PCAUtility
from image_utility import ImageUtility
import numpy as np
from train import Train
from test import Test

if __name__ == '__main__':
    tf_record_util = TFRecordUtility()
    pca_utility = PCAUtility()
    cnn_model = CNNModel()
    image_utility = ImageUtility()

    # trainer = Train(use_tf_record=False,
    #                 dataset_name=DatasetName.ibug,
    #                 custom_loss=False,
    #                 arch='mn_main',
    #                 inception_mode=True,
    #                 num_output_layers=1,
    #                 point_wise=True,
    #                 weight=None
    #                 )

    # tester = Test()

    # trainer = Train(use_tf_record=True,
    #                 dataset_name=DatasetName.ibug,
    #                 custom_loss=False,
    #                 arch='mn_r',
    #                 inception_mode=False,
    #                 num_output_layers=1,
    #                 weight=None)
    # tf_record_util.create_fused_images_and_labels_name()

    trainer = Train(use_tf_record=False,
                    dataset_name=DatasetName.ibug,
                    custom_loss=False,
                    arch='sh_reg',
                    inception_mode=False,
                    num_output_layers=1,
                    weight=None,
                    point_wise=True)






