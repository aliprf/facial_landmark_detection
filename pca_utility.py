from configuration import DatasetName, DatasetType,\
    AffectnetConf, IbugConf, W300Conf, InputDataSize, LearningConfig
from tf_record_utility import TFRecordUtility

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pickle


class PCAUtility:
    __eigenvalues_prefix = "_eigenvalues_"
    __eigenvectors_prefix = "_eigenvectors_"
    __meanvector_prefix = "_meanvector_"

    def create_pca(self, dataset_name, pca_postfix):
        tf_record_util = TFRecordUtility()

        lbl_arr = []
        pose_arr = []
        if dataset_name == DatasetName.ibug:
            lbl_arr, img_arr, pose_arr = tf_record_util.retrieve_tf_record(IbugConf.tf_train_path,
                                                                 IbugConf.sum_of_train_samples,
                                                                 only_label=True, only_pose=True)
        lbl_arr = np.array(lbl_arr)

        print('PCA-retrieved')

        '''need to be normalized based on the hyper face paper?'''

        # reduced_lbl_arr, eigenvalues, eigenvectors = self.__svd_func(lbl_arr, pca_postfix)
        reduced_lbl_arr, eigenvalues, eigenvectors = self.__func_PCA(lbl_arr, pca_postfix)
        mean_lbl_arr = np.mean(lbl_arr, axis=0)
        eigenvectors = eigenvectors.T

        self.__save_obj(eigenvalues, dataset_name + self.__eigenvalues_prefix + str(pca_postfix))
        self.__save_obj(eigenvectors, dataset_name + self.__eigenvectors_prefix + str(pca_postfix))
        self.__save_obj(mean_lbl_arr, dataset_name + self.__meanvector_prefix + str(pca_postfix))

        '''calculate pose min max'''
        p_1_arr = []
        p_2_arr = []
        p_3_arr = []

        for p_item in pose_arr:
            p_1_arr.append(p_item[0])
            p_2_arr.append(p_item[1])
            p_3_arr.append(p_item[2])

        p_1_min = min(p_1_arr)
        p_1_max = max(p_1_arr)

        p_2_min = min(p_2_arr)
        p_2_max = max(p_2_arr)

        p_3_min = min(p_3_arr)
        p_3_max = max(p_3_arr)

        self.__save_obj(p_1_min, 'p_1_min')
        self.__save_obj(p_1_max, 'p_1_max')

        self.__save_obj(p_2_min, 'p_2_min')
        self.__save_obj(p_2_max, 'p_2_max')

        self.__save_obj(p_3_min, 'p_3_min')
        self.__save_obj(p_3_max, 'p_3_max')

        print('PCA-->done')

    def __save_obj(self, obj, name):
        with open('obj/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_pose_obj(self):
        with open('obj/p_1_min.pkl', 'rb') as f:
            p_1_min = pickle.load(f)
        with open('obj/p_1_max.pkl', 'rb') as f:
            p_1_max = pickle.load(f)

        with open('obj/p_2_min.pkl', 'rb') as f:
            p_2_min = pickle.load(f)
        with open('obj/p_2_max.pkl', 'rb') as f:
            p_2_max = pickle.load(f)

        with open('obj/p_3_min.pkl', 'rb') as f:
            p_3_min = pickle.load(f)
        with open('obj/p_3_max.pkl', 'rb') as f:
            p_3_max = pickle.load(f)

        return p_1_min, p_1_max, p_2_min, p_2_max, p_3_min, p_3_max


    def load_pca_obj(self, dataset_name, pca_postfix=97):
        with open('obj/' + dataset_name + self.__eigenvalues_prefix + str(pca_postfix) + '.pkl', 'rb') as f:
            eigenvalues = pickle.load(f)
        with open('obj/' + dataset_name + self.__eigenvectors_prefix + str(pca_postfix) + '.pkl', 'rb') as f:
            eigenvectors = pickle.load(f)
        with open('obj/' + dataset_name + self.__meanvector_prefix + str(pca_postfix) + '.pkl', 'rb') as f:
            meanvector = pickle.load(f)
        return eigenvalues, eigenvectors, meanvector

    def __func_PCA(self, input_data, pca_postfix):
        input_data = np.array(input_data)
        pca = PCA(n_components=pca_postfix/100)
        # pca = PCA(n_components=0.98)
        # pca = IncrementalPCA(n_components=50, batch_size=50)
        pca.fit(input_data)
        pca_input_data = pca.transform(input_data)
        eigenvalues = pca.explained_variance_
        eigenvectors = pca.components_
        return pca_input_data, eigenvalues, eigenvectors

    def __svd_func(self, input_data, pca_postfix):
        svd = TruncatedSVD(n_components=50)
        svd.fit(input_data)
        pca_input_data = svd.transform(input_data)
        eigenvalues = svd.explained_variance_
        eigenvectors = svd.components_
        return pca_input_data, eigenvalues, eigenvectors
        # U, S, VT = svd(input_data)


