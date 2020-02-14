from configuration import DatasetName, DatasetType, \
    AffectnetConf, IbugConf, W300Conf, InputDataSize, LearningConfig
from tf_record_utility import TFRecordUtility
from pca_utility import PCAUtility
from image_utility import ImageUtility
from pose_detection.code.PoseDetector import PoseDetector
from pose_detection.code import datasets, hopenet, hopelessnet, utils
from hg_Class import HourglassNet

import tensorflow as tf
import keras
from skimage.transform import resize

print(tf.__version__)
print(keras.__version__)

from keras.regularizers import l2
tf.logging.set_verbosity(tf.logging.ERROR)
from keras.models import Model
from keras.applications import mobilenet_v2, mobilenet, resnet50, densenet
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, \
    BatchNormalization, Activation, GlobalAveragePooling2D, DepthwiseConv2D, Dropout, ReLU, Concatenate, Deconvolution2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from keras.optimizers import adam
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.callbacks import CSVLogger
from datetime import datetime

import  cv2
import os.path
from keras.utils.vis_utils import plot_model
from scipy.spatial import distance
import scipy.io as sio
from keras.preprocessing.image import ImageDataGenerator
from clr_callback import CyclicLR

class Test:
    def test_new(self, arch):
        # detect = PoseDetector()
        detect = 0

        tf_record_utility = TFRecordUtility()
        image_utility = ImageUtility()
        lbl_arr_challenging, img_arr_challenging = tf_record_utility.retrieve_tf_record_test_set(
            tfrecord_filename=W300Conf.tf_challenging,
            number_of_records=W300Conf.number_of_all_sample_challenging,
            only_label=False)
        lbl_arr_common, img_arr_common = tf_record_utility.retrieve_tf_record_test_set(
            tfrecord_filename=W300Conf.tf_common,
            number_of_records=W300Conf.number_of_all_sample_common,
            only_label=False)
        lbl_arr_full, img_arr_full = tf_record_utility.retrieve_tf_record_test_set(tfrecord_filename=W300Conf.tf_full,
                                                                                   number_of_records=W300Conf.number_of_all_sample_full,
                                                                                   only_label=False)

        lbl_arr_challenging = np.array(lbl_arr_challenging)
        img_arr_challenging = np.array(img_arr_challenging)

        lbl_arr_common = np.array(lbl_arr_common)
        img_arr_common = np.array(img_arr_common)

        lbl_arr_full = np.array(lbl_arr_full)
        img_arr_full = np.array(img_arr_full)

        if arch == 'mn':  # main mobileNet
            model = self.mnv2_hm(tensor=None)
        elif arch == 'mn_r':  # reduced mobileNet
            model = self.mnv2_hm_r_v2(tensor=None, inception_mode= False)

        model.load_weights("weights-02-0.00021.h5")  # no:   ** asm:

        ########

        loss_challenging = 0

        loss_yaw_ch = 0
        loss_yaw_c = 0
        loss_yaw_f = 0
        loss_pitch_ch = 0
        loss_pitch_c = 0
        loss_pitch_f = 0
        loss_roll_ch = 0
        loss_roll_c = 0
        loss_roll_f = 0

        loss_common = 0
        loss_full = 0
        # multitask = True
        all_true = []
        all_pridicted  = []
        for i in range(W300Conf.number_of_all_sample_challenging):
            loss_challenging_, mae_yaw, mae_pitch, mae_roll, lt, lp = self._test_result_per_image(i, model,
                                                                                         img_arr_challenging[i],
                                                                                         lbl_arr_challenging[i],
                                                                                         detect)
            loss_challenging += loss_challenging_
            loss_yaw_ch += mae_yaw
            loss_pitch_ch += mae_pitch
            loss_roll_ch += mae_roll

            all_true.append(lt)
            all_pridicted.append(lp)

        sio.savemat('all_true.mat', {'ground_truth_all': np.array(all_true)})
        sio.savemat('all_pridicted.mat', {'detected_points_all': np.array(all_pridicted)})
        print('LOSS challenging: ')

        print(loss_challenging * 100 / W300Conf.number_of_all_sample_challenging)
        print(loss_yaw_ch / W300Conf.number_of_all_sample_challenging)
        print(loss_pitch_ch / W300Conf.number_of_all_sample_challenging)
        print(loss_roll_ch / W300Conf.number_of_all_sample_challenging)

        for i in range(W300Conf.number_of_all_sample_common):
            loss_common_, mae_yaw, mae_pitch, mae_roll, _,_ = self._test_result_per_image(i, model, img_arr_common[i],
                                                                                    lbl_arr_common[i], detect)
            loss_common += loss_common_
            loss_yaw_c += mae_yaw
            loss_pitch_c += mae_pitch
            loss_roll_c += mae_roll

        print('LOSS common: ')
        print(loss_common * 100 / W300Conf.number_of_all_sample_common)
        print(loss_yaw_c / W300Conf.number_of_all_sample_common)
        print(loss_pitch_c / W300Conf.number_of_all_sample_common)
        print(loss_roll_c / W300Conf.number_of_all_sample_common)

        for i in range(W300Conf.number_of_all_sample_full):
            loss_full_, mae_yaw, mae_pitch, mae_roll, _,_ = self._test_result_per_image(i, model, img_arr_full[i],
                                                                                  lbl_arr_full[i], detect)
            loss_full += loss_full_
            loss_yaw_f += mae_yaw
            loss_pitch_f += mae_pitch
            loss_roll_f += mae_roll

        print('LOSS full: ')
        print(loss_full * 100 / W300Conf.number_of_all_sample_full)
        print(loss_yaw_f / W300Conf.number_of_all_sample_full)
        print(loss_pitch_f / W300Conf.number_of_all_sample_full)
        print(loss_roll_f / W300Conf.number_of_all_sample_full)

    def _test_result_per_image(self, counter, model, img, labels_true, detect):
        tf_utility = TFRecordUtility()
        image_utility = ImageUtility()

        pose_predicted = []
        image = np.expand_dims(img, axis=0)

        predict = model.predict(image)

        heatmap_main = predict[0]
        # heatmap_main = predict[0][0]

        # heatmap_main_all = predict[1][0]
        # labels_predicted = np.swapaxes(predict[2], 0, 1)
        # pose_predicted = predict[3][0]


        # print("labels_true: " + str(labels_true))
        # print("labels_predicted :" + str(labels_predicted))

        labels_true_transformed, landmark_arr_x_t, landmark_arr_y_t = image_utility. \
            create_landmarks_from_normalized(labels_true, 224, 224, 112, 112)
        #
        # labels_predicted_asm = self.__post_process_correction(labels_predicted)

        # labels_predict_transformed, landmark_arr_x_p, landmark_arr_y_p = image_utility. \
            # create_landmarks_from_normalized(labels_predicted, 224, 224, 112, 112)

        # labels_predict_asm_transformed, landmark_arr_x_asm_p, landmark_arr_y_asm_p = image_utility. \
        #     create_landmarks_from_normalized(labels_predicted_asm, 224, 224, 112, 112)

        # xys = np.zeros(shape=[136])
        # xys[72:96] = labels_true_transformed[72:96]
        # # xys[73] = labels_true_transformed[73]
        # # xys[90] = labels_true_transformed[90]
        # # xys[91] = labels_true_transformed[91]
        #
        # x_s =[]
        # x_y =[]
        # for i in range(0, 136, 2):
        #     x_s.append(xys[i])
        #     x_y.append(xys[i+1])
        # imgpr.print_image_arr(counter+1, img, x_s, x_y)

        x_h_p, y_h_p, xy_h_p = tf_utility.from_heatmap_to_point(heatmap_main, 5)

        # mkps = tf_utility.get_predicted_kp_from_htmap(heatmap_main, (112, 112), 1, (56, 56)) # (68 * 3)
        # x_h_p = []
        # y_h_p = []
        # xy_h_p = []
        # for i in range(mkps.shape[0]):
        #     x_h_p.append(mkps[i][0])
        #     y_h_p.append(mkps[i][1])
        #
        #     xy_h_p.append(mkps[i][0])
        #     xy_h_p.append(mkps[i][1])

        labels_predict_transformed = xy_h_p

        '''asm pp'''
        # xy_h_p_asm = self.__post_process_correction(xy_h_p, True)
        # labels_predict_transformed_asm, landmark_arr_x_p_asm, landmark_arr_y_p_asm = image_utility. \
        #     create_landmarks_from_normalized(xy_h_p_asm, 224, 224, 112, 112)
        # labels_predict_transformed = labels_predict_transformed_asm
        ''''''

        #

        # heatmap_main_all = heatmap_main_all[:, :, 0]

        # print(heatmap_main_all.shape)
        # imgpr.print_image_arr(counter+1, heatmap_main_all, np.array(landmark_arr_x_t)/4, np.array(landmark_arr_y_t)/4)
        # imgpr.print_image_arr(counter+1, heatmap_main_all, np.array(x_h_p)/4, np.array(y_h_p)/4)

        # imgpr.print_image_arr((counter+1)*100, img, x_h_p, y_h_p)

        # imgpr.print_image_arr(counter+1, img, landmark_arr_x_p_asm, landmark_arr_y_p_asm)

        # imgpr.print_image_arr((counter+1)*1000, img, landmark_arr_x_p, landmark_arr_y_p)

        # imgpr.print_image_arr_heat(counter+1, heatmap_main, print_single=False)
        #
        # imgpr.print_image_arr((counter+1)*100000, img, landmark_arr_x_t, landmark_arr_y_t)


        # print("landmark_arr_x_t: " + str(landmark_arr_x_t))
        # print("landmark_arr_x_p :" + str(landmark_arr_x_p))
        #
        # print("landmark_arr_y_t: " + str(landmark_arr_y_t))
        # print("landmark_arr_y_p :" + str(landmark_arr_y_p))

        # return 0,0,0,0

        interpupil_distance = self.__calculate_interpupil_distance(labels_true_transformed)
        # interpupil_distance = self.__calculate_interoccular_distance(labels_true_transformed)

        sum_errors = 0
        for i in range(0, len(labels_true_transformed), 2): # two step each time
            '''this is the value after transformation to the real points'''
            x_point_predicted = labels_predict_transformed[i]
            y_point_predicted = labels_predict_transformed[i+1]

            # x_point_predicted_asm = labels_predict_asm_transformed[i]
            # y_point_predicted_asm = labels_predict_asm_transformed[i+1]
            #

            x_point_true = labels_true_transformed[i]
            y_point_true = labels_true_transformed[i+1]

            '''this is the normalized value, which predicted by network'''
            error = math.sqrt(((x_point_predicted - x_point_true) ** 2) + ((y_point_predicted - y_point_true) ** 2))
            sum_errors += error

        normalized_mean_error = sum_errors / (interpupil_distance * (LearningConfig.landmark_len/2))
        # print(normalized_mean_error)
        # print('=====')

        lp = np.array(labels_predict_transformed).reshape([68, 2])
        lt = np.array(labels_true_transformed).reshape([68, 2])

        # print(labels_true_transformed)
        # print(lt)
        # print('---------------')

        return normalized_mean_error, 0, 0, 0, lt, lp

        '''pose estimation vs hopeNet'''
        img_cp_1 = np.array(img) * 255.0
        r, g, b = cv2.split(img_cp_1)
        img_cp_1 = cv2.merge([b, g, r])

        img_cp_2 = np.array(img) * 255.0
        r, g, b = cv2.split(img_cp_2)
        img_cp_2 = cv2.merge([b, g, r])

        # yaw_truth, pitch_truth, roll_truth = 0, 0, 0
        yaw_truth, pitch_truth, roll_truth = detect.detect(img, isFile=False, show=False)

        yaw_p = pose_predicted[0]
        pitch_p = pose_predicted[1]
        roll_p = pose_predicted[2]
        ''' normalized to normal '''
        min_degree = - 65
        max_degree = 65
        yaw_tpre = min_degree + (max_degree - min_degree) * (yaw_p+1)/2
        pitch_tpre = min_degree + (max_degree - min_degree) * (pitch_p+1)/2
        roll_tpre = min_degree + (max_degree - min_degree) * (roll_p+1)/2

        # print("true:  " + str(yaw_truth)+"--"+str(pitch_truth)+"--"+str(roll_truth))
        # print("predict:  " + str(yaw_tpre)+"--"+str(pitch_tpre)+"--"+str(roll_tpre))

        # output_pre = utils.draw_axis(img_cp_1, yaw_tpre, pitch_tpre, roll_tpre, tdx=150, tdy=150, size=150)
        # output_truth = utils.draw_axis(img_cp_2, yaw_truth, pitch_truth, roll_truth, tdx=150, tdy=150, size=150)
        # cv2.imwrite(str(counter+1) + ".jpg", output_pre)
        # cv2.imwrite(str((counter+1)*1000) + ".jpg", output_truth)

        mae_yaw = abs(yaw_tpre - yaw_truth)
        mae_pitch = abs(pitch_tpre - pitch_truth)
        mae_roll = abs(roll_tpre - roll_truth)
        "================================="
        return normalized_mean_error, mae_yaw, mae_pitch, mae_roll

    def __calculate_interoccular_distance(self, labels_true):
        left_oc_x = labels_true[72]
        left_oc_y = labels_true[73]

        right_oc_x = labels_true[90]
        right_oc_y = labels_true[91]

        distance = math.sqrt(((left_oc_x - right_oc_x) ** 2) + ((left_oc_y - right_oc_y) ** 2))
        return distance

    def __calculate_interpupil_distance(self, labels_true):
        # points: x,y 36--> 41 point for left, and 42->47 for right

        left_pupil_x = (labels_true[72]+labels_true[74]+labels_true[76]+labels_true[78]+labels_true[80]+labels_true[82])/6
        left_pupil_y = (labels_true[73]+labels_true[75]+labels_true[77]+labels_true[79]+labels_true[81]+labels_true[83])/6

        right_pupil_x = (labels_true[84] + labels_true[86] + labels_true[88] + labels_true[90] + labels_true[92] + labels_true[94]) / 6
        right_pupil_y = (labels_true[85] + labels_true[87] + labels_true[89] + labels_true[91] + labels_true[93] + labels_true[95]) / 6

        dis = math.sqrt(((left_pupil_x - right_pupil_x)**2) + ((left_pupil_y - right_pupil_y)**2))

        # p1 = [left_pupil_x, left_pupil_y]
        # p2 = [right_pupil_x, right_pupil_y]
        # dis1 = distance.euclidean(p1, p2)
        #
        # print(dis)
        # print(dis1)
        # print('==============') both are equal
        return dis

    def init_for_test(self):
        self.init_tensors(test=True)
        # create video:
        # video = cv2.VideoWriter("baby.avi", 0, 25.0, (720, 486))
        # f_arr = os.listdir("./image_baby/")
        # f_arr.sort()
        # for file in f_arr:
        #     f_name = os.path.join("./image_baby/", file)
        #     img = cv2.imread(f_name)
        #     video.write(img)
        #
        # cv2.destroyAllWindows()
        # video.release()
        # return
        #

        # model = self.__shrink_v3_mobileNet_v2_multi_task(tensor=None, pose=True)
        model = self.__shrink_v6_mobileNet_v2_single_task(tensor=None, pose=True,test=True)
        model.load_weights("weights-05-0.05.h5")  # no: 0.19980816035713   ** asm: 0.20618409347179825  weights-01-0.01
        # model.load_weights("weight_deepASMNet_asm.h5")  # no: 0.19980816035713   ** asm: 0.20618409347179825  weights-01-0.01
        model.summary()

        png_file_arr = []
        file_box_arr = []
        i = 0
        video = cv2.VideoWriter("output.avi", 0, 25.0, (1280, 720))

        f_arr = os.listdir("../tensorflow_face_detection/img/")
        f_arr.sort()
        for file in f_arr:
            if file.endswith(".jpg") or file.endswith(".png"):
                ff = file.split("-")[1]
                y1, x1, y2, x2 = ff[:-4].split("__")
                file_box_arr.append([y1, x1, y2, x2])
                f_name = os.path.join("../tensorflow_face_detection/img/", file)
                png_file_arr.append(f_name)
                img = cv2.imread(f_name)

                crop_img = img[int(float(y1)):int(float(y2)), int(float(x1)):int(float(x2))]
                # cv2.imwrite("cropped"+file, crop_img)

                resized_img = resize(crop_img,
                                     (InputDataSize.image_input_size, InputDataSize.image_input_size, 3),
                                     anti_aliasing=True)

                scale_x = crop_img.shape[1] / resized_img.shape[1]
                scale_y = crop_img.shape[0] / resized_img.shape[0]

                out_img = self.test_per_image(i, model, resized_img, True, 0,  img, x1, y1, scale_x, scale_y)
                video.write(out_img)
                print("v..")
                i += 1

        cv2.destroyAllWindows()
        video.release()

    def test_per_image(self, counter, model, img, multitask, detect, org_img, x1, y1, scale_x, scale_y):

        image_utility = ImageUtility()
        pose_predicted = []
        image = np.expand_dims(img, axis=0)
        if multitask:
            labels_main = np.swapaxes(model.predict(image)[0], 0, 1)
            # noise_lbls = model.predict(image)[1]
            # eyes_lbls = model.predict(image)[2]
            # face_lbls = model.predict(image)[3]
            # mouth_lbls = model.predict(image)[4]
            pose_predicted = model.predict(image)[1][0]
            # lbls_partial = np.swapaxes(np.concatenate((face_lbls, noise_lbls, eyes_lbls, mouth_lbls), axis=1), 0, 1)  #

            labels_predicted = labels_main
            # labels_predicted = lbls_partial
            # labels_predicted = (lbls_partial + labels_main) / 2.0
        else:
            labels_predicted = np.swapaxes(model.predict(image), 0, 1)

        # labels_true_transformed, landmark_arr_x_t, landmark_arr_y_t = image_utility. \
        #     create_landmarks_from_normalized(labels_true, 224, 224, 112, 112)
        #
        # labels_predicted_asm = self.__post_process_correction(labels_predicted)

        # labels_predict_asm_transformed, landmark_arr_x_asm_p, landmark_arr_y_asm_p = image_utility. \
        #     create_landmarks_from_normalized(labels_predicted_asm, 224, 224, 112, 112)

        # imgpr.print_image_arr(counter+1, img, [], [])
        # imgpr.print_image_arr(counter+1, img, landmark_arr_x_p, landmark_arr_y_p)

        # imgpr.print_image_arr(counter+1, org_img, landmark_arr_x_p, landmark_arr_y_p)

        # imgpr.print_image_arr((counter+1)*1000, img, landmark_arr_x_t, landmark_arr_y_t)
        # imgpr.print_image_arr((counter+1)*100000, img, landmark_arr_x_asm_p, landmark_arr_y_asm_p)


        '''pose estimation vs hopeNet'''
        img_cp_1 = np.array(img) * 255.0
        r, g, b = cv2.split(img_cp_1)
        img_cp_1 = cv2.merge([b, g, r])

        img_cp_2 = np.array(img) * 255.0
        r, g, b = cv2.split(img_cp_2)
        img_cp_2 = cv2.merge([b, g, r])

        # yaw_truth, pitch_truth, roll_truth = 0, 0, 0
        yaw_truth, pitch_truth, roll_truth = detect.detect(img, isFile=False, show=False)

        yaw_p = pose_predicted[0]
        pitch_p = pose_predicted[1]
        roll_p = pose_predicted[2]
        ''' normalized to normal '''
        min_degree = - 65
        max_degree = 65
        yaw_tpre = min_degree + (max_degree - min_degree) * (yaw_p+1)/2
        pitch_tpre = min_degree + (max_degree - min_degree) * (pitch_p+1)/2
        roll_tpre = min_degree + (max_degree - min_degree) * (roll_p+1)/2

        output_pre = utils.draw_axis(org_img, yaw_tpre, pitch_tpre, roll_tpre, tdx=float(x1)+112, tdy=float(y1)+112, size=112)

        labels_predict_transformed, landmark_arr_x_p, landmark_arr_y_p, img_cv2 = image_utility. \
            create_landmarks_from_normalized_original_img(output_pre, labels_predicted, 224, 224, 112, 112, float(x1),
                                                          float(y1), scale_x, scale_y)

        # cv2.imwrite(str(counter) + ".jpg", output_pre, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        mae_yaw = abs(yaw_tpre - yaw_truth)
        mae_pitch = abs(pitch_tpre - pitch_truth)
        mae_roll = abs(roll_tpre - roll_truth)
        "================================="
        return output_pre
