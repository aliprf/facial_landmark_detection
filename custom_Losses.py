from configuration import DatasetName, DatasetType, \
    AffectnetConf, IbugConf, W300Conf, InputDataSize, LearningConfig
from tf_record_utility import TFRecordUtility
from cnn_model import CNNModel
from pca_utility import PCAUtility
from image_utility import ImageUtility
from data_Heatmap_Generator import DataHeatmapGenerator
import tensorflow as tf
import keras

tf.logging.set_verbosity(tf.logging.ERROR)
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import math

import  cv2
import os.path
from keras.utils.vis_utils import plot_model
from scipy.spatial import distance
import scipy.io as sio


class Custom_losses:
    def _calculate_mse(self, y_p, y_t):
        mse = (np.square(y_p - y_t)).mean(axis=None)
        # print('y_p: '+str(y_p.shape))
        # print('mse: '+str(mse.shape))
        # loss = 0
        # for j in range(len(y_p)):
        #     loss += (y_p[j] - y_t[j]) ** 2
        # loss /= len(y_p)

        # print('calculate_mse: ' + str(mse))
        return mse

    def _generate_distance_matrix(self, xy_arr):
        x_arr = xy_arr[[slice(None, None, 2) for _ in range(xy_arr.ndim)]]
        y_arr = xy_arr[[slice(1, None, 2) for _ in range(xy_arr.ndim)]]

        d_matrix = np.zeros(shape=[len(x_arr), len(y_arr)])
        for i in range(0, x_arr.shape[0], 1):
            for j in range(i + 1, x_arr.shape[0], 1):
                p1 = [x_arr[i], y_arr[i]]
                p2 = [x_arr[j], y_arr[j]]
                d_matrix[i, j] = distance.euclidean(p1, p2)
                d_matrix[j, i] = distance.euclidean(p1, p2)
        return d_matrix

    def _depart_facial_point(self, xy_arr):
        face = xy_arr[0:54]  # landmark_face_len = 54
        nose = xy_arr[54:72]  # landmark_nose_len = 18
        leys = xy_arr[72:84]  # landmark_eys_len = 24
        reys = xy_arr[84:96]  # landmark_eys_len = 24
        mouth = xy_arr[96:136]  # landmark_mouth_len = 40
        return face, nose, leys, reys, mouth

    def custom_loss_hm(self, ten_hm_t, ten_hm_p):
        # print(ten_hm_t.get_shape().as_list())  #  [None, 56, 56, 68]
        # print(ten_hm_p.get_shape())

        tf_utility = TFRecordUtility()

        sqr = K.square(ten_hm_t - ten_hm_p)  # [None, 56, 56, 68]
        mean1 = K.mean(sqr, axis=1)
        mean2 = K.mean(mean1, axis=1)
        tensor_mean_square_error = K.mean(mean2, axis=1)
        # print(tensor_mean_square_error.get_shape().as_list())  # [None, 68]

        # vec_mse = K.eval(tensor_mean_square_error)
        # print("mse.shape:")
        # print(vec_mse.shape)  # (50, 68)
        # print(vec_mse)
        # print("----------->>>")

        '''convert tensor to vector'''
        vec_hm_p = K.eval(ten_hm_p)
        vec_hm_t = K.eval(ten_hm_t)

        loss_array = []

        for i in range(LearningConfig.batch_size):
            '''convert heatmap to points'''
            x_h_p, y_h_p, xy_h_p = tf_utility.from_heatmap_to_point(vec_hm_p[i], 5, 1)
            x_h_t, y_h_t, xy_h_t = tf_utility.from_heatmap_to_point(vec_hm_t[i], 5, 1)

            '''normalise points to be in [0, 1]'''
            x_h_p = x_h_p / 56
            y_h_p = y_h_p / 56
            xy_h_p = xy_h_p / 56
            x_h_t = x_h_t / 56
            y_h_t = y_h_t / 56
            xy_h_t = xy_h_t / 56
            '''test print images'''
            # imgpr.print_image_arr(i + 1, np.zeros(shape=[56, 56]), x_h_t, y_h_t)
            # imgpr.print_image_arr((i + 1)*1000, np.zeros(shape=[56, 56]), x_h_p, y_h_p)

            # print('--xy_h_p:---')
            # print(xy_h_p)
            # print('--xy_h_t:---')
            # print(xy_h_t)

            face_p, mouth_p, nose_p, leye_p, reye_p = self._depart_facial_point(xy_h_p)
            face_t, mouth_t, nose_t, leye_t, reye_t = self._depart_facial_point(xy_h_t)

            '''generate facial distance matrix'''
            face_p_mat, face_t_mat = self._generate_distance_matrix(face_p), self._generate_distance_matrix(face_t)
            mouth_p_mat, mouth_t_mat = self._generate_distance_matrix(mouth_p), self._generate_distance_matrix(mouth_t)
            nose_p_mat, nose_t_mat = self._generate_distance_matrix(nose_p), self._generate_distance_matrix(nose_t)
            leye_p_mat, leye_t_mat = self._generate_distance_matrix(leye_p), self._generate_distance_matrix(leye_t)
            reye_p_mat, reye_t_mat = self._generate_distance_matrix(reye_p), self._generate_distance_matrix(reye_t)

            '''calculate loss from each pair matrices'''

            face_loss = LearningConfig.reg_term_face * self._calculate_mse(face_p_mat, face_t_mat) / len(face_p)
            mouth_loss = LearningConfig.reg_term_mouth * self._calculate_mse(mouth_p_mat, mouth_t_mat) / len(mouth_p)
            nose_loss = LearningConfig.reg_term_nose * self._calculate_mse(nose_p_mat, nose_t_mat) / len(nose_p)
            leye_loss = LearningConfig.reg_term_leye * self._calculate_mse(leye_p_mat, leye_t_mat) / len(leye_p)
            reye_loss = LearningConfig.reg_term_reye * self._calculate_mse(reye_p_mat, reye_t_mat) / len(reye_p)

            loss_array.append(face_loss + mouth_loss + nose_loss + leye_loss + reye_loss)

            # print('mse[i]: ' + str(vec_mse[i]))
            # print('face_loss[i]: ' + str(face_loss))
            # print('mouth_loss[i]: ' + str(mouth_loss))
            # print('nose_loss[i]: ' + str(nose_loss))
            # print('leye_loss[i]: ' + str(leye_loss))
            # print('reye_loss[i]: ' + str(reye_loss))
            # print('============')

        loss_array = np.array(loss_array)
        tensor_distance_loss = K.variable(loss_array)

        # tensor_total_loss = tf.reduce_mean([tensor_mean_square_error, loss_array])
        tensor_total_loss = tf.add(tensor_mean_square_error, tensor_distance_loss)
        return tensor_total_loss

    def __inceptionLoss_1(self, yTrue, yPred):
        return self.__soft_MSE(yTrue, yPred, 20)

    def __inceptionLoss_2(self, yTrue, yPred):
        return self.__soft_MSE(yTrue, yPred, 10)

    def __inceptionLoss_3(self, yTrue, yPred):
        return self.__soft_MSE(yTrue, yPred, 5)

    def __soft_MSE(self,  yTrue, yPred, boundary_count, radius=0.01):
        yTrue_vector_batch = K.eval(yTrue)
        yPred_vector_batch = K.eval(yPred)

        out_batch_vector = []  # 50 *136
        for i in range(LearningConfig.batch_size):
            out_vector = []  # 136
            for j in range(LearningConfig.batch_size):
                if abs(yTrue_vector_batch[i,j] - yPred_vector_batch[i,j]) <= boundary_count * radius:
                    out_vector.append(0)
                else:
                    out_vector.append(1)
            out_batch_vector.append(out_vector)

        out_batch_vector = np.array(out_batch_vector)
        out_batch_tensor = K.variable(out_batch_vector)

        tensor_mean_square_error = K.mean(K.square(yPred - yTrue), axis=-1)
        tmp_mul = tf.multiply(tensor_mean_square_error, out_batch_tensor)
        return tmp_mul

    def __ASM(self, input_tensor, pca_postfix):
        print(pca_postfix)
        pca_utility = PCAUtility()
        image_utility = ImageUtility()
        eigenvalues, eigenvectors, meanvector = pca_utility.load_pca_obj(DatasetName.ibug, pca_postfix=pca_postfix)

        input_vector_batch = K.eval(input_tensor)
        out_asm_vector = []
        for i in range(LearningConfig.batch_size):
            b_vector_p = self.calculate_b_vector(input_vector_batch[i], True, eigenvalues, eigenvectors, meanvector)
            # asm_vector = meanvector + np.dot(eigenvectors, b_vector_p)
            #
            # labels_predict_transformed, landmark_arr_x_p, landmark_arr_y_p = \
            #     image_utility.create_landmarks_from_normalized(asm_vector, 224, 224, 112, 112)
            # imgpr.print_image_arr(i + 1, np.zeros(shape=[224,224,3]), landmark_arr_x_p, landmark_arr_y_p)

            out_asm_vector.append(meanvector + np.dot(eigenvectors, b_vector_p))

        out_asm_vector = np.array(out_asm_vector)

        tensor_out = K.variable(out_asm_vector)
        return tensor_out

    def __customLoss(self, yTrue, yPred):
        pca_utility = PCAUtility()
        eigenvalues, eigenvectors, meanvector = pca_utility.load_pca_obj(DatasetName.ibug)

        # yTrue = tf.constant([[1.0, 2.0, 3.0], [5.0, 4.0, 7.0]])
        # yPred = tf.constant([[2.0, 5.0, 6.0], [7.0, 3.0, 8.0]])
        # session = K.get_session()
        bias = 1
        tensor_mean_square_error = K.log((K.mean(K.square(yPred - yTrue), axis=-1) + bias))
        mse = K.eval(tensor_mean_square_error)
        # print("mse:")
        # print(mse)
        # print("---->>>")

        yPred_arr = K.eval(yPred)
        yTrue_arr = K.eval(yTrue)

        loss_array = []

        for i in range(LearningConfig.batch_size):
            asm_loss = 0

            truth_vector = yTrue_arr[i]
            predicted_vector = yPred_arr[i]

            b_vector_p = self.calculate_b_vector(predicted_vector, True, eigenvalues, eigenvectors, meanvector)
            y_pre_asm = meanvector + np.dot(eigenvectors, b_vector_p)

            for j in range(len(y_pre_asm)):
                asm_loss += (truth_vector[j] - y_pre_asm[j]) ** 2
            asm_loss /= len(y_pre_asm)

            asm_loss += bias + 1
            asm_loss = math.log(asm_loss, 10)
            asm_loss *= LearningConfig.regularization_term
            loss_array.append(asm_loss)
            print('mse[i]' + str(mse[i]))
            print('asm_loss[i]' + str(asm_loss))
            print('============')

        loss_array = np.array(loss_array)

        tensor_asm_loss = K.variable(loss_array)
        tensor_total_loss = tf.reduce_mean([tensor_mean_square_error, tensor_asm_loss], axis=0)

        return tensor_total_loss

    def __customLoss_base(self, yTrue, yPred):
        pca_utility = PCAUtility()
        image_utility = ImageUtility()
        tf_record_utility = TFRecordUtility()

        eigenvalues, eigenvectors, meanvector = pca_utility.load_pca_obj(DatasetName.ibug)

        # yTrue = tf.constant([[1.0, 2.0, 3.0], [5.0, 4.0, 7.0]])
        # yPred = tf.constant([[9.0, 1.0, 2.0], [7.0, 3.0, 8.0]])
        # session = K.get_session()

        tensor_mean_square_error = K.mean(K.square(yPred - yTrue), axis=-1)
        # tensor_mean_square_error = keras.losses.mean_squared_error(yPred, yTrue)
        mse = K.eval(tensor_mean_square_error)

        yPred_arr = K.eval(yPred)
        yTrue_arr = K.eval(yTrue)

        loss_array = []

        for i in range(LearningConfig.batch_size):
            asm_loss = 0

            truth_vector = yTrue_arr[i]
            predicted_vector = yPred_arr[i]

            b_vector_p = self.calculate_b_vector(predicted_vector, True, eigenvalues, eigenvectors, meanvector)
            y_pre_asm = meanvector + np.dot(eigenvectors, b_vector_p)

            """in order to test the results after PCA, you can use these lines of code"""
            # landmark_arr_xy, landmark_arr_x, landmark_arr_y = image_utility.create_landmarks_from_normalized(truth_vector, 224, 224, 112, 112)
            # image_utility.print_image_arr(i, np.ones([224, 224]), landmark_arr_x, landmark_arr_y)
            #
            # landmark_arr_xy_new, landmark_arr_x_new, landmark_arr_y_new= image_utility.create_landmarks_from_normalized(y_pre_asm, 224, 224, 112, 112)
            # image_utility.print_image_arr(i*100, np.ones([224, 224]), landmark_arr_x_new, landmark_arr_y_new)

            for j in range(len(y_pre_asm)):
                asm_loss += (truth_vector[j] - y_pre_asm[j]) ** 2
            asm_loss /= len(y_pre_asm)


            # asm_loss *= mse[i]
            # asm_loss *= LearningConfig.regularization_term

            loss_array.append(asm_loss)

            print('mse[i]' + str(mse[i]))
            print('asm_loss[i]' + str(asm_loss))
            print('============' )

        loss_array = np.array(loss_array)
        tensor_asm_loss = K.variable(loss_array)

        # sum_loss_tensor = tf.math.add(tensor_mean_square_error, tensor_asm_loss)
        tensor_total_loss = tf.reduce_mean([tensor_mean_square_error, tensor_asm_loss], axis=0)

        # sum_loss = np.array(K.eval(tensor_asm_loss))
        # print(mse)
        # print(K.eval(tensor_mean_square_error))
        # print(K.eval(tensor_asm_loss))
        # print('asm_loss  ' + str(loss_array[0]))
        # print('mse_loss  ' + str(mse[0]))
        # print('sum_loss  ' + str(sum_loss[0]))
        # print('total_loss  ' + str(total_loss[0]))
        # print('      ')
        return tensor_total_loss

    def init_tensors(self, test):
        batchsize = LearningConfig.batch_size
        if test:
            batchsize = 1

        pca_utility = PCAUtility()
        eigenvalues, eigenvectors, meanvector = pca_utility.load_pca_obj(DatasetName.ibug, )

        # print("predicted_tensor " + str(predicted_tensor.shape))
        # print("meanvector " + str(meanvector.shape))
        # print("eigenvalues " + str(eigenvalues.shape))
        # print("eigenvectors " + str(eigenvectors.shape))
        # print("-")

        self._meanvector_arr = np.tile(meanvector, (batchsize, 1))
        # meanvector_arr = np.tile(meanvector[None, :, None], (LearningConfig.batch_size, 1, 1))
        # print("meanvector_arr" + str(meanvector_arr.shape))

        self._eigenvalues_arr = np.tile(eigenvalues, (batchsize, 1))
        # eigenvalues_arr = np.tile(eigenvalues[None, :, None], (LearningConfig.batch_size, 1, 1))
        # print("eigenvalues_arr" + str(eigenvalues_arr.shape))

        self._eigenvectors_arr = np.tile(eigenvectors[None, :, :], (batchsize, 1, 1))
        # print("eigenvectors_arr" + str(eigenvectors_arr.shape))

        self._meanvector_tensor = tf.convert_to_tensor(self._meanvector_arr, dtype=tf.float32)
        self._eigenvalues_tensor = tf.convert_to_tensor(self._eigenvalues_arr, dtype=tf.float32)
        self._eigenvectors_tensor = tf.convert_to_tensor(self._eigenvectors_arr, dtype=tf.float32)

        self._eigenvectors_T = tf.transpose(self._eigenvectors_tensor, perm=[0, 2, 1])
        print("")

    def custom_activation(self, predicted_tensor):
        pca_utility = PCAUtility()
        eigenvalues, eigenvectors, meanvector = pca_utility.load_pca_obj(DatasetName.ibug)

        b_vector_tensor = self.calculate_b_vector_tensor(predicted_tensor, True, eigenvalues,
                                                         self._eigenvectors_tensor, self._meanvector_tensor)

        out = tf.add(tf.expand_dims(self._meanvector_tensor, 2), tf.matmul(self._eigenvectors_tensor, b_vector_tensor))
        out = tf.reshape(out, [LearningConfig.batch_size, 136])

        return out

    def calculate_b_vector_tensor(self, predicted_tensor, correction, eigenvalues, eigenvectors, mean_tensor):
        tmp1 = tf.expand_dims(tf.subtract(predicted_tensor, mean_tensor), 2)

        b_vector_tensor = tf.matmul(self._eigenvectors_T, tmp1)  # (50, 50, 1)

        return b_vector_tensor

        b_vector = np.squeeze(K.eval(b_vector_tensor), axis=2)[0]
        print("b_vector -> " + str(b_vector.shape))  # (50,)

        mul_arr = np.ones(b_vector.shape)
        add_arr = np.zeros(b_vector.shape)

        # put b in -3lambda =>
        if correction:
            i = 0
            for b_item in b_vector:
                lambda_i_sqr = 3 * math.sqrt(eigenvalues[i])

                if b_item > 0:
                    if b_item > lambda_i_sqr:
                        mul_arr[i] = 0.0
                        add_arr[i] = lambda_i_sqr
                    b_item = min(b_item, lambda_i_sqr)
                else:
                    if b_item < -1 * lambda_i_sqr:
                        mul_arr[i] = 0.0
                        add_arr[i] = lambda_i_sqr
                    b_item = max(b_item, -1 * lambda_i_sqr)

                b_vector[i] = b_item
                i += 1

        mul_arr = np.tile(mul_arr, (LearningConfig.batch_size, 1))
        add_arr = np.tile(add_arr, (LearningConfig.batch_size, 1))

        # print(mul_arr)
        # print(add_arr)

        mul_tensor = tf.expand_dims(tf.convert_to_tensor(mul_arr, dtype=tf.float32), 2)
        add_arr = tf.expand_dims(tf.convert_to_tensor(add_arr, dtype=tf.float32), 2)

        # print("mul_tensor -> " + str(mul_tensor.shape))  # (50, 50, 1)
        # print("add_arr -> " + str(add_arr.shape))  # (50, 50, 1)

        tmp_mul = tf.multiply(b_vector_tensor, mul_tensor)
        tmp_add = tf.add(tmp_mul, add_arr)

        # print("add_arr -> " + str(add_arr.shape))  # (50, 50, 1)

        return tmp_add

    def custom_activation_test(self, predicted_tensor):
        pca_utility = PCAUtility()
        eigenvalues, eigenvectors, meanvector = pca_utility.load_pca_obj(DatasetName.ibug)

        b_vector_tensor = self.calculate_b_vector_tensor_test(predicted_tensor, True, eigenvalues,
                                                              self._eigenvectors_tensor, self._meanvector_tensor)

        out = tf.add(tf.expand_dims(self._meanvector_tensor, 2), tf.matmul(self._eigenvectors_tensor, b_vector_tensor))
        out = tf.reshape(out, [1, 136])

        return out

    def calculate_b_vector_tensor_test(self, predicted_tensor, correction, eigenvalues, eigenvectors, mean_tensor):

        # print("predicted_tensor -> " + str(predicted_tensor.shape))  # (50,)
        # print("eigenvalues -> " + str(eigenvalues.shape))  # (50,)
        # print("eigenvectors -> " + str(eigenvectors.shape))  # (50,)
        # print("mean_tensor -> " + str(mean_tensor.shape))  # (50,)

        tmp1 = tf.expand_dims(tf.math.subtract(predicted_tensor, mean_tensor), 2)
        # print("tmp1 -> " + str(tmp1.shape))  # (50,)

        b_vector_tensor = tf.matmul(self._eigenvectors_T, tmp1)  # (50, 50, 1)
        print("b_vector_tensor -> " + str(b_vector_tensor.shape))  # (50,)
        return b_vector_tensor

        # inputs = K.placeholder(shape=(None, 224, 224, 3))
        #
        # sess = K.get_session()
        # tmp22 = sess.run(inputs)
        # tmp22 = K.get_value(b_vector_tensor)

        tmp22 = K.eval(b_vector_tensor)

        # holder = tf.placeholder(tf.float32, shape=(None, 224,224,3))
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     print("12")
        #     tmp22 = sess.run([b_vector_tensor], feed_dict=holder)
        # b_vector_tensor.eval(feed_dict=holder)

        print("tmp22 -> " + str(tmp22.shape))
        b_vector = np.squeeze(K.eval(b_vector_tensor), axis=2)
        print("b_vector -> " + str(b_vector.shape))  # (50,)

        mul_arr = np.ones(b_vector.shape)
        add_arr = np.zeros(b_vector.shape)

        # put b in -3lambda =>
        if correction:
            i = 0
            for b_item in b_vector:
                lambda_i_sqr = 3 * math.sqrt(eigenvalues[i])

                if b_item > 0:
                    if b_item > lambda_i_sqr:
                        mul_arr[i] = 0.0
                        add_arr[i] = lambda_i_sqr
                    b_item = min(b_item, lambda_i_sqr)
                else:
                    if b_item < -1 * lambda_i_sqr:
                        mul_arr[i] = 0.0
                        add_arr[i] = lambda_i_sqr
                    b_item = max(b_item, -1 * lambda_i_sqr)

                b_vector[i] = b_item
                i += 1

        mul_arr = np.tile(mul_arr, (1, 1))
        add_arr = np.tile(add_arr, (1, 1))

        # print(mul_arr)
        # print(add_arr)

        mul_tensor = tf.expand_dims(tf.convert_to_tensor(mul_arr, dtype=tf.float32), 2)
        add_arr = tf.expand_dims(tf.convert_to_tensor(add_arr, dtype=tf.float32), 2)

        # print("mul_tensor -> " + str(mul_tensor.shape))  # (50, 50, 1)
        # print("add_arr -> " + str(add_arr.shape))  # (50, 50, 1)

        tmp_mul = tf.math.multiply(b_vector_tensor, mul_tensor)
        tmp_add = tf.math.add(tmp_mul, add_arr)

        print("tmp_add -> " + str(tmp_add.shape))  # (50, 50, 1)

        return tmp_add

    def calculate_b_vector(self, predicted_vector, correction, eigenvalues, eigenvectors, meanvector):
        tmp1 = predicted_vector - meanvector
        b_vector = np.dot(eigenvectors.T, tmp1)

        # put b in -3lambda =>
        if correction:
            i = 0
            for b_item in b_vector:
                lambda_i_sqr = 3 * math.sqrt(eigenvalues[i])

                if b_item > 0:
                    b_item = min(b_item, lambda_i_sqr)
                else:
                    b_item = max(b_item, -1 * lambda_i_sqr)
                b_vector[i] = b_item
                i += 1

        return b_vector

    def __reorder(self, input_arr):
        out_arr = []
        for i in range(68):
            out_arr.append(input_arr[i])
            k = 68 + i
            out_arr.append(input_arr[k])
        return np.array(out_arr)

    def test_pca_validity(self, pca_postfix):
        cnn_model = CNNModel()
        pca_utility = PCAUtility()
        tf_record_utility = TFRecordUtility()
        image_utility = ImageUtility()

        eigenvalues, eigenvectors, meanvector = pca_utility.load_pca_obj(dataset_name=DatasetName.ibug, pca_postfix=pca_postfix)

        lbl_arr, img_arr, pose_arr = tf_record_utility.retrieve_tf_record(tfrecord_filename=IbugConf.tf_train_path,
                                                                          number_of_records=30, only_label=False)
        for i in range(20):
            b_vector_p = cnn_model.calculate_b_vector(lbl_arr[i], True, eigenvalues, eigenvectors, meanvector)
            lbl_new = meanvector + np.dot(eigenvectors, b_vector_p)

            labels_true_transformed, landmark_arr_x_t, landmark_arr_y_t = image_utility.\
                create_landmarks_from_normalized(lbl_arr[i], 224, 224, 112, 112)

            labels_true_transformed_pca, landmark_arr_x_pca, landmark_arr_y_pca = image_utility. \
                create_landmarks_from_normalized(lbl_new, 224, 224, 112, 112)

            image_utility.print_image_arr(i, img_arr[i], landmark_arr_x_t, landmark_arr_y_t)
            image_utility.print_image_arr(i * 1000, img_arr[i], landmark_arr_x_pca, landmark_arr_y_pca)

    _meanvector_arr = []
    _eigenvalues_arr = []
    _eigenvectors_arr = []
    _meanvector_tensor = None
    _eigenvalues_tensor = None
    _eigenvectors_tensor = None
    _eigenvectors_T = None