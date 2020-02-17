from tf_record_utility import TFRecordUtility
from configuration import DatasetName, DatasetType, AffectnetConf, IbugConf, W300Conf, InputDataSize
from cnn_model import CNNModel
from pca_utility import PCAUtility
from image_utility import ImageUtility
import numpy as np
from train import Train
if __name__ == '__main__':
    tf_record_util = TFRecordUtility()
    pca_utility = PCAUtility()
    cnn_model = CNNModel()
    image_utility = ImageUtility()

    # tf_record_util.generate_hm_and_save()
    # tf_record_util.retrive_hm_and_test()

    # mat = np.random.randint(0, 10, size=10)
    # cnn_model.generate_distance_matrix(mat)

    # hm = np.random.randint(0, 10, size=(10, 10, 68))
    # tf_record_util.from_heatmap_to_point(hm, 5)
    #
    """creating tf_records file"""
    # tf_record_util.create_tf_record(dataset_name=DatasetName.ibug, dataset_type=None,
    #                                 thread_number=None, number_of_threads=None, heatmap=True)

    # tf_record_util.create_tf_record(dataset_name=DatasetName.w300, dataset_type=0,  # challenging
    #                                 thread_number=None, number_of_threads=None)
    # tf_record_util.create_tf_record(dataset_name=DatasetName.w300, dataset_type=1,  # common
    #                                 thread_number=None, number_of_threads=None)
    # tf_record_util.create_tf_record(dataset_name=DatasetName.w300, dataset_type=2,  # full
    #                                 thread_number=None, number_of_threads=None)
    # cnn_model.test_result(multitask=True, singletask = True)


    # cnn_model.test_result(multitask=True, singletask=False, pose=True)
    # cnn_model.test_result(multitask=False, singletask=True, pose=True)
    # cnn_model.test_result_distilation()

    # cnn_model.init_for_test()

    #
    # """creating tf_records file"""
    # tf_record_util.create_tf_record(dataset_name=DatasetName.w300, dataset_type=None,
    #                                 thread_number=None, number_of_threads=None)

    # tf_record_util.create_tf_record(dataset_name=DatasetName.aflw, dataset_type=None,
    #                                 thread_number=None, number_of_threads=None)


    """creating PCA from tf_records"""
    # pca_postfix = 95  # 80 85 90 95
    # pca_utility.create_pca(dataset_name=DatasetName.ibug, pca_postfix=pca_postfix)
    # cnn_model.test_pca_validity(pca_postfix=pca_postfix)

    # cnn_model.train_multi_task(dataset_name=DatasetName.ibug, asm_loss=False, pose=True)

    # cnn_model.train_multi_task(dataset_name=DatasetName.ibug, asm_loss=True, pose=True)

    # cnn_model.train_multi_task_inception(dataset_name=DatasetName.ibug, pose=True)

    # cnn_model.train_distil_model_inception_heat(dataset_name=DatasetName.ibug)

    # cnn_model.train_distil_model_inception_reg_heat_v0(dataset_name=DatasetName.ibug)
    # cnn_model.test_distil_model_reg_heat_inception()

    # cnn_model.train_distil_model_reg_heat(dataset_name=DatasetName.ibug)
    # cnn_model.test_distil_model_reg_heat()
    # cnn_model.test_distil_model_reg_heat_inception()

    # cnn_model.train(dataset_name=DatasetName.ibug, asm_loss=True)

    # cnn_model.train_new(dataset_name=DatasetName.ibug, custom_loss=False, arch='mn_r', inception_mode=True) # mn, mn_r
    # cnn_model.test_new(arch='mn_r') # mn, mn_r

    trainer = Train(use_tf_record=False,
                    dataset_name=DatasetName.ibug,
                    custom_loss=False,
                    arch='hg',
                    inception_mode=True,
                    num_output_layers=2,
                    weight=None)






