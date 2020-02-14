class DatasetName:
    affectnet = 'affectnet'
    w300 = 'w300'
    ibug = 'ibug'
    aflw = 'aflw'
    aflw2000 = 'aflw2000'



class DatasetType:
    data_type_train = 0
    data_type_validation = 1
    data_type_test = 2


class LearningConfig:
    weight_loss_heatmap_face = 4.0
    weight_loss_heatmap_all_face = 4.0
    weight_loss_regression_face = 2.0
    weight_loss_regression_pose = 1.0

    weight_loss_heatmap_face_inc1 = 0.5
    weight_loss_heatmap_all_face_inc1 = 0.5
    weight_loss_regression_face_inc1 = 0.25
    weight_loss_regression_pose_inc1 = 0.125

    weight_loss_heatmap_face_inc2 = 1.0
    weight_loss_heatmap_all_face_inc2 = 1.0
    weight_loss_regression_face_inc2 = 0.5
    weight_loss_regression_pose_inc2 = 0.25

    weight_loss_heatmap_face_inc3 = 1.5
    weight_loss_heatmap_all_face_inc3 = 1.5
    weight_loss_regression_face_inc3 = 0.75
    weight_loss_regression_pose_inc3 = 0.375

    loss_weight_inception_1_face = 2
    # loss_weight_inception_1_pose = 1

    loss_weight_inception_2_face = 5
    # loss_weight_inception_2_pose = 2

    loss_weight_inception_3_face = 8
    # loss_weight_inception_3_pose = 3

    loss_weight_pose = 0.5

    loss_weight_face = 1
    loss_weight_nose = 1
    loss_weight_eyes = 1
    loss_weight_mouth = 1

    CLR_METHOD = "triangular"
    MIN_LR = 1e-7
    MAX_LR = 1e-2
    STEP_SIZE = 10

    batch_size = 50
    steps_per_validation_epochs = 5
    epochs = 150
    landmark_len = 136

    reg_term_face = 5.0  # 0.9
    reg_term_mouth = 10.0  # 0.9
    reg_term_nose = 10.0  # 0.9
    reg_term_leye = 10.0  # 0.9
    reg_term_reye = 10.0  # 0.9



class InputDataSize:
    image_input_size = 224
    landmark_len = 136
    landmark_face_len = 54
    landmark_nose_len = 18
    landmark_eys_len = 24
    landmark_mouth_len = 40
    pose_len = 3


class AffectnetConf:
    csv_train_path = '/media/ali/extradata/facial_landmark_ds/affectNet/training.csv'
    csv_evaluate_path = '/media/ali/extradata/facial_landmark_ds/affectNet/validation.csv'
    csv_test_path = '/media/ali/extradata/facial_landmark_ds/affectNet/test.csv'

    tf_train_path = '/media/ali/extradata/facial_landmark_ds/affectNet/train.tfrecords'
    tf_test_path = '/media/ali/extradata/facial_landmark_ds/affectNet/eveluate.tfrecords'
    tf_evaluation_path = '/media/ali/extradata/facial_landmark_ds/affectNet/test.tfrecords'

    sum_of_train_samples = 200000  # 414800
    sum_of_test_samples = 30000
    sum_of_validation_samples = 5500

    img_path_prefix = '/media/ali/extradata/facial_landmark_ds/affectNet/Manually_Annotated_Images/'


class Multipie:
    lbl_path_prefix = '/media/ali/extradata/facial_landmark_ds/multi-pie/MPie_Labels/labels/all/'
    img_path_prefix = '/media/ali/extradata/facial_landmark_ds/multi-pie/'

    origin_number_of_all_sample = 2000
    origin_number_of_train_sample = 1950
    origin_number_of_evaluation_sample = 50
    augmentation_factor = 100


class W300Conf:
    tf_common = '/media/ali/extradata/facial_landmark_ds/from_ibug/test_set/test_common.tfrecords'
    tf_challenging = '/media/ali/extradata/facial_landmark_ds/from_ibug/test_set/test_challenging.tfrecords'
    tf_full = '/media/ali/extradata/facial_landmark_ds/from_ibug/test_set/test_full.tfrecords'

    img_path_prefix_common = '/media/ali/extradata/facial_landmark_ds/from_ibug/test_set/common/'
    img_path_prefix_challenging = '/media/ali/extradata/facial_landmark_ds/from_ibug/test_set/challenging/'
    img_path_prefix_full = '/media/ali/extradata/facial_landmark_ds/from_ibug/test_set/full/'

    number_of_all_sample_common = 554
    number_of_all_sample_challenging = 135
    number_of_all_sample_full = 689


class IbugConf:
    tf_train_path = '/media/ali/extradata/facial_landmark_ds/from_ibug/train_set/train.tfrecords'
    tf_test_path = '/media/ali/extradata/facial_landmark_ds/from_ibug/train_set/test.tfrecords'
    tf_evaluation_path = '/media/ali/extradata/facial_landmark_ds/from_ibug/train_set/evaluation.tfrecords'

    tf_train_path_heatmap = '/media/ali/extradata/facial_landmark_ds/from_ibug/train_set/train_heatmap.tfrecords'
    tf_test_path_heatmap = '/media/ali/extradata/facial_landmark_ds/from_ibug/train_set/test_heatmap.tfrecords'
    tf_evaluation_path_heatmap = '/media/ali/extradata/facial_landmark_ds/from_ibug/train_set/evaluation_heatmap.tfrecords'

    # origin_number_of_all_sample = 3148  # afw, train_helen, train_lfpw
    # origin_number_of_train_sample = 2834  # 95 % for train
    # origin_number_of_evaluation_sample = 314  # 5% for evaluation

    origin_number_of_all_sample = 33672  # afw, train_helen, train_lfpw
    origin_number_of_train_sample = 31989  # 95 % for train
    origin_number_of_evaluation_sample = 1683  # 5% for evaluation

    augmentation_factor = 10  # create 100 image from 1
    augmentation_factor_rotate = 20  # create 100 image from 1

    sum_of_train_samples = origin_number_of_train_sample * augmentation_factor
    sum_of_validation_samples = origin_number_of_evaluation_sample * augmentation_factor

    img_path_prefix = '/media/ali/extradata/facial_landmark_ds/from_ibug/train_set/all/'

    rotated_img_path_prefix = '/media/ali/extradata/facial_landmark_ds/from_ibug/train_set/train_rotated/'
    before_heatmap_img_path_prefix = '/media/ali/extradata/facial_landmark_ds/from_ibug/train_set/train_before_heatmap/'
