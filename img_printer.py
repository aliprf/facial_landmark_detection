import matplotlib.pyplot as plt
from pathlib import Path


def print_image(image_name, landmarks_x, landmarks_y):

    my_file = Path(image_name)
    if my_file.is_file():
        im = plt.imread(image_name)
        implot = plt.imshow(im)

        plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='r', s=10)
        plt.show()


def print_image_arr_heat(k, image, print_single=False):
    import numpy as np
    for i in range(image.shape[2]):
        img = np.sum(image, axis=2)
        if print_single:
            plt.figure()
            plt.imshow(image[:, :, i])
            implot = plt.imshow(image[:, :, i])
            plt.axis('off')
            plt.savefig('single_heat_' + str(i+(k*100)) + '.png', bbox_inches='tight')
            plt.clf()

    plt.figure()
    plt.imshow(img)
    implot = plt.imshow(img)
    plt.axis('off')
    plt.savefig('heat_' + str(k) + '.png', bbox_inches='tight')
    plt.clf()


def print_image_arr(k, image, landmarks_x, landmarks_y):
    plt.figure()
    plt.imshow(image)
    implot = plt.imshow(image)

    # for i in range(len(landmarks_x)):
    #     plt.text(landmarks_x[i], landmarks_y[i], str(i), fontsize=12, c='red',
    #              horizontalalignment='center', verticalalignment='center',
    #              bbox={'facecolor': 'blue', 'alpha': 0.3, 'pad': 0.0})

    plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='black', s=20)
    plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='white', s=15)
    plt.axis('off')
    plt.savefig('sss' + str(k) + '.png', bbox_inches='tight')
    # plt.show()
    plt.clf()


def print_image_arr_2(k, image, landmarks_x, landmarks_y, xs, ys):
    plt.figure()
    plt.imshow(image)
    implot = plt.imshow(image)

    # plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='b', s=5)
    # for i in range(68):
    #     plt.annotate(str(i), (landmarks_x[i], landmarks_y[i]), fontsize=6)

    plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='b', s=5)
    plt.scatter(x=xs, y=ys, c='r', s=5)
    plt.savefig('sss'+str(k)+'.png')
    # plt.show()
    plt.clf()


def print_two_landmarks(image, landmarks_1, landmarks_2):
    plt.figure()
    plt.imshow(image)
    implot = plt.imshow(image)

    plt.scatter(x=landmarks_1[:68], y=landmarks_1[68:], c='b', s=10)
    plt.scatter(x=landmarks_2[:68], y=landmarks_2[68:], c='r', s=10)
    # plt.savefig('a'+str(landmarks_x[0])+'.png')
    plt.show()
    plt.clf()