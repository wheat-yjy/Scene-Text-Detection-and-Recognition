import tensorflow as tf
import itertools
import numpy as np
import os
import keras
import pickle
import cv2
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Input, Dense, BatchNormalization, Flatten, Softmax
from keras.layers import Reshape, Lambda
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
OUTPUT_DIR = 'AON_2'
BATCH_SIZE = 1
run_name = "2019:05:08:09:09:38"

# pickle a variable to a file TODO PATH!
file = open('text_recognize/label_map_ft.pickle', 'rb+')
label_map = pickle.load(file)

TEST_FUNC = None


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, :, :]
    input_length = keras.backend.reshape(input_length, (BATCH_SIZE, 1))
    label_length = keras.backend.reshape(label_length, (BATCH_SIZE, 1))
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def text_to_labels(text):
    ret = []
    for char in text:
        ret.append(label_map.find(char))
    return ret


# Reverse translation of numerical classes back to characters
def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == 0:  # CTC Blank
            ret.append("")
        else:
            for (key, value) in label_map.items():
                if value == c:
                    ret.append(key)
    return "".join(ret)


def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 0:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ret


def squeeze(x):
    return K.squeeze(x, axis=1)


def concat(args):
    a, b = args
    return tf.concat((a, b), axis=2)


def concat_1(args):
    a, b = args
    return tf.concat((a, b), axis=1)


def reverse(a):
    return tf.reverse(a, axis=[1])


def transpose(a):
    return tf.transpose(a, perm=[0, 2, 1])


def filter_gate(args):
    HN_gru_merged, HN_gru_merged_reverse, VN_gru_merged, VN_gru_merged_reverse, cpc_out = args

    A = HN_gru_merged * tf.tile(tf.reshape(cpc_out[:, :, 0], [-1, 23, 1]), [1, 1, 512])
    B = HN_gru_merged_reverse * tf.tile(tf.reshape(cpc_out[:, :, 1], [-1, 23, 1]), [1, 1, 512])
    C = VN_gru_merged * tf.tile(tf.reshape(cpc_out[:, :, 2], [-1, 23, 1]), [1, 1, 512])
    D = VN_gru_merged_reverse * tf.tile(tf.reshape(cpc_out[:, :, 3], [-1, 23, 1]), [1, 1, 512])
    res = A + B + C + D
    res = keras.activations.tanh(res)

    return res


def rot90(a):
    return tf.image.rot90(a)


def regular_rectangle(img, orig, width, height):

    """
    :param image: just the image
    :param orig: orig is a list which contains 4 points's coordinates in the dataset
    :param width: the new image's width
    :param height: the new image's height
    :return: the image which has been converted to a rectangel not a polygon
    """
    # 左上 左下 右下 右上
    pts1 = np.float32([[orig[1], orig[2], orig[0], orig[3]]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (width, height))
    return dst


def BCNN(input_data):
    # BCNN
    inner = ZeroPadding2D(padding=(1, 1))(input_data)
    inner = Conv2D(64, (3, 3), padding="valid",
                   activation='relu', kernel_initializer='he_normal',
                   name='conv1')(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2), strides=[2, 2], padding='valid', name='max1')(inner)

    inner = ZeroPadding2D(padding=(1, 1))(inner)
    inner = Conv2D(128, (3, 3), padding='valid',
                   activation='relu', kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = BatchNormalization()(inner)
    inner = ZeroPadding2D(padding=(1, 1))(inner)
    inner = MaxPooling2D(pool_size=(2, 2), strides=[2, 2], padding='valid', name='max2')(inner)

    inner = ZeroPadding2D(padding=(1, 1))(inner)
    inner = Conv2D(256, (3, 3), strides=[1, 1], padding='valid',
                   activation='relu', kernel_initializer='he_normal',
                   name='conv3')(inner)
    inner = BatchNormalization()(inner)

    inner = ZeroPadding2D(padding=(1, 1))(inner)
    inner = Conv2D(256, (3, 3), strides=[1, 1], padding='valid',
                   activation='relu', kernel_initializer='he_normal',
                   name='conv4')(inner)
    inner = BatchNormalization()(inner)

    return inner


def load_model(model_name):
    # Input Parameters
    input_data = Input(name='the_input', shape=[BATCH_SIZE, 100, 100, 3], dtype='float32',
                       batch_shape=[BATCH_SIZE, 100, 100, 3])

    inner = BCNN(input_data)

    # AON set shared_conv function
    shared_conv1 = Conv2D(512, (3, 3), padding='valid',
                          activation='relu', kernel_initializer='he_normal',
                          name='shared_conv1')
    shared_pool1 = MaxPooling2D(pool_size=(2, 2), strides=[2, 1], padding='valid', name='shared_max1')

    shared_conv2 = Conv2D(512, (3, 3), padding='valid',
                          activation='relu', kernel_initializer='he_normal',
                          name='shared_conv2')
    shared_pool2 = MaxPooling2D(pool_size=(2, 2), strides=[2, 1], padding='valid', name='shared_max2')

    shared_conv3 = Conv2D(512, (3, 3), padding='valid',
                          activation='relu', kernel_initializer='he_normal',
                          name='shared_conv3')
    shared_pool3 = MaxPooling2D(pool_size=(2, 2), strides=[2, 1], padding='valid', name='shared_max3')

    shared_conv4 = Conv2D(512, (3, 3), padding='valid',
                          activation='relu', kernel_initializer='he_normal',
                          name='shared_conv4')
    shared_pool4 = MaxPooling2D(pool_size=(2, 2), strides=[2, 1], padding='valid', name='shared_max4')

    shared_conv5 = Conv2D(512, (3, 3), padding='valid',
                          activation='relu', kernel_initializer='he_normal',
                          name='shared_conv5')
    shared_pool5 = MaxPooling2D(pool_size=(2, 2), strides=[2, 1], padding='valid', name='shared_max5')

    # AON -- HN
    HN_inner = ZeroPadding2D(padding=(1, 1))(inner)
    HN_inner = shared_conv1(HN_inner)
    HN_inner = BatchNormalization()(HN_inner)
    HN_inner = ZeroPadding2D(padding=(1, 0))(HN_inner)
    HN_inner = shared_pool1(HN_inner)

    HN_inner = ZeroPadding2D(padding=(1, 1))(HN_inner)
    HN_inner = shared_conv2(HN_inner)
    HN_inner = BatchNormalization()(HN_inner)
    HN_inner = ZeroPadding2D(padding=(0, 1))(HN_inner)
    HN_inner = shared_pool2(HN_inner)

    HN_inner = ZeroPadding2D(padding=(1, 1))(HN_inner)
    HN_inner = shared_conv3(HN_inner)
    HN_inner = ZeroPadding2D(padding=(1, 0))(HN_inner)
    HN_inner = BatchNormalization()(HN_inner)
    HN_inner = shared_pool3(HN_inner)

    HN_inner = ZeroPadding2D(padding=(1, 1))(HN_inner)
    HN_inner = shared_conv4(HN_inner)
    HN_inner = BatchNormalization()(HN_inner)
    HN_inner = shared_pool4(HN_inner)

    HN_inner = ZeroPadding2D(padding=(1, 1))(HN_inner)
    HN_inner = shared_conv5(HN_inner)
    HN_inner = BatchNormalization()(HN_inner)
    HN_inner = shared_pool5(HN_inner)

    HN_inner = Lambda(squeeze)(HN_inner)

    HN_gru = GRU(256, return_sequences=True,
                 kernel_initializer='he_normal', name='HN_gru')(HN_inner)
    HN_gru_b = GRU(256, return_sequences=True,
                   go_backwards=True, kernel_initializer='he_normal',
                   name='HN_gru_b')(HN_inner)
    HN_gru_merged = Lambda(concat)([HN_gru, HN_gru_b])
    #     HN_gru_merged = tf.concat((HN_gru, HN_gru_b), axis=2)
    HN_gru_merged_reverse = Lambda(reverse)(HN_gru_merged)
    #     HN_gru_merged_reverse = tf.reverse(HN_gru_merged, axis=[1])

    # AON -- VN
    # rotate
    VN_inner = Lambda(rot90)(inner)

    VN_inner = ZeroPadding2D(padding=(1, 1))(VN_inner)
    VN_inner = shared_conv1(VN_inner)
    VN_inner = BatchNormalization()(VN_inner)
    VN_inner = ZeroPadding2D(padding=(1, 0))(VN_inner)
    VN_inner = shared_pool1(VN_inner)

    VN_inner = ZeroPadding2D(padding=(1, 1))(VN_inner)
    VN_inner = shared_conv2(VN_inner)
    VN_inner = BatchNormalization()(VN_inner)
    VN_inner = ZeroPadding2D(padding=(0, 1))(VN_inner)
    VN_inner = shared_pool2(VN_inner)

    VN_inner = ZeroPadding2D(padding=(1, 1))(VN_inner)
    VN_inner = shared_conv3(VN_inner)
    VN_inner = BatchNormalization()(VN_inner)
    VN_inner = ZeroPadding2D(padding=(1, 0))(VN_inner)
    VN_inner = shared_pool3(VN_inner)

    VN_inner = ZeroPadding2D(padding=(1, 1))(VN_inner)
    VN_inner = shared_conv4(VN_inner)
    VN_inner = BatchNormalization()(VN_inner)
    VN_inner = shared_pool4(VN_inner)

    VN_inner = ZeroPadding2D(padding=(1, 1))(VN_inner)
    VN_inner = shared_conv5(VN_inner)
    VN_inner = BatchNormalization()(VN_inner)
    VN_inner = shared_pool5(VN_inner)

    VN_inner = Lambda(squeeze)(VN_inner)

    VN_gru = GRU(256, return_sequences=True,
                 kernel_initializer='he_normal', name='VN_gru')(VN_inner)
    VN_gru_b = GRU(256, return_sequences=True,
                   go_backwards=True, kernel_initializer='he_normal',
                   name='VN_gru_b')(VN_inner)
    VN_gru_merged = Lambda(concat)([VN_gru, VN_gru_b])
    VN_gru_merged_reverse = Lambda(reverse)(VN_gru_merged)

    # AON -- character_placement_clues
    cpc_inner = ZeroPadding2D(padding=(1, 1))(inner)
    cpc_inner = Conv2D(512, (3, 3), padding='valid',
                       activation='relu', kernel_initializer='he_normal',
                       name='cpc_conv1')(cpc_inner)
    cpc_inner = BatchNormalization()(cpc_inner)
    cpc_inner = ZeroPadding2D(padding=(1, 1))(cpc_inner)
    cpc_inner = MaxPooling2D(pool_size=(2, 2), strides=[2, 2], padding='valid', name='cpc_max1')(cpc_inner)

    cpc_inner = ZeroPadding2D(padding=(1, 1))(cpc_inner)
    cpc_inner = Conv2D(512, (3, 3), padding='valid',
                       activation='relu', kernel_initializer='he_normal',
                       name='cpc_conv2')(cpc_inner)
    cpc_inner = BatchNormalization()(cpc_inner)
    cpc_inner = ZeroPadding2D(padding=(1, 1))(cpc_inner)
    cpc_inner = MaxPooling2D(pool_size=(2, 2), strides=[2, 2], padding='valid', name='cpc_max2')(cpc_inner)

    cpc_inner = Flatten()(cpc_inner)
    cpc_inner = Reshape(target_shape=[64, 512])(cpc_inner)
    cpc_inner = Lambda(transpose)(cpc_inner)
    #     cpc_inner = tf.transpose(cpc_inner, perm=[0, 2, 1])

    cpc_inner = Dense(units=23, activation='relu', name='cpc_dense1')(cpc_inner)
    cpc_inner = Reshape(target_shape=[512, 23])(cpc_inner)
    cpc_inner = Lambda(transpose)(cpc_inner)
    #     cpc_inner = tf.transpose(cpc_inner, perm=[0, 2, 1])

    cpc_inner = Dense(units=4, activation='relu', name='cpc_dense2')(cpc_inner)
    cpc_inner = Reshape(target_shape=[23, 4])(cpc_inner)
    cpc_out = Softmax(axis=2, name='softmax')(cpc_inner)

    res = Lambda(filter_gate)([HN_gru_merged, HN_gru_merged_reverse, VN_gru_merged, VN_gru_merged_reverse, cpc_out])

    # print("res.shape:\t{}".format(res.shape))

    decoder_gru = GRU(256, return_sequences=True,
                      kernel_initializer='he_normal', name='decoder_gru', )(res)
    decoder_gru_b = GRU(256, return_sequences=True,
                        go_backwards=True, kernel_initializer='he_normal',
                        name='decoder_gru_b')(res)
    decoder_merged = Lambda(concat_1)([decoder_gru, decoder_gru_b])

    y_pred = Dense(len(label_map) + 2, activation='softmax', kernel_initializer='he_normal',
                   name='dense_out')(decoder_merged)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels',
                   shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=5e-4, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_data, labels, input_length, label_length],
                  outputs=loss_out,
                  )

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_pred, decoder_out: decoder_out}, optimizer=sgd)

    weight_file = 'text_recognize/weights%02d.h5' % (390)
    model.load_weights(weight_file)

    # captures output of softmax so we can decode the output during visualization

    test_func = K.function([input_data], [y_pred])
    global TEST_FUNC
    if TEST_FUNC is None:
        TEST_FUNC = test_func
    # index = 0
    # while index < 16:
    #     word_batch = next(img_gen.next_train())[0]
    #     # print(word_batch)
    #     decoded_res = decode_batch(test_func,
    #                                word_batch['the_input'])
    #     index += 1
    #     for i in range(len(word_batch)):
    #         print("origin:\t{}, predict:\t{}".format(labels_to_text(word_batch['the_labels'][i]), decoded_res[i]))


def one_image(image, text_positions):  # [[[1,1], [2,2], [2,0], [1,0]]]
    '''
    识别单张图片
    :param img:
    :param text_position: [[左上，右上，右下，左下], []]
    :return:
    '''


    results = []

    for point in text_positions:
        min_x = min([int(float(point[0][0])), int(float(point[1][0])), int(float(point[2][0])), int(float(point[3][0]))])
        max_x = max([int(float(point[0][0])), int(float(point[1][0])), int(float(point[2][0])), int(float(point[3][0]))])
        min_y = min([int(float(point[0][1])), int(float(point[1][1])), int(float(point[2][1])), int(float(point[3][1]))])
        max_y = max([int(float(point[0][1])), int(float(point[1][1])), int(float(point[2][1])), int(float(point[3][1]))])
        single_regular_area = image[min_y:max_y, min_x:max_x]
        single_regular_area = cv2.resize(single_regular_area, (100, 100), interpolation=cv2.INTER_CUBIC)

        # print("看这里这里有坐标\t {} \t".format(points)) # 我弄的。。
        X_data = np.reshape(single_regular_area, [BATCH_SIZE, 100, 100, 3])

        global TEST_FUNC
        text = TEST_FUNC([X_data])[0]
        # print(text)
        ret = []

        for j in range(text.shape[0]):
            out_best = list(np.argmax(text[j, 0:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            outstr = labels_to_text(out_best)
            ret.append(outstr)
        print("识别结果为:{}".format(ret))

        single_item = {}
        single_item['x0'] = min_x
        single_item['y0'] = min_y
        single_item['x1'] = max_x
        single_item['y1'] = max_y
        single_item['text'] = ret[0]
        results.append(single_item)

        # cv2.imwrite("test.jpg", single_regular_area)
        # regular_text_images.append(single_regular_area)
    print(results)
    return results



def test(index):
    image_file_name = "/home/benke/veteranjy/data/images/tr_img_" + str(index) + ".jpg"
    # image_file_name = "/home/benke/xuyichu/PSENet/tmp/tr_img_03001.jpg"
    img = cv2.imread(image_file_name)

    label_file_name = "/home/benke/veteranjy/data/labels/tr_img_" + str(index) + ".txt"
    # label_file_name = "/home/benke/xuyichu/PSENet/tmp/tr_img_03001.txt"

    position_labels = []
    word_labels = []

    lines = open(label_file_name, encoding='utf-8')
    for line in lines:
        elements = line.split(',')
        word = elements[-1].replace('\n', '')
        word_labels.append(word)

        bottom_left = [int(float(elements[0])), int(float(elements[1]))]
        bottom_right = [int(float(elements[2])), int(float(elements[3]))]
        top_right = [int(float(elements[4])), int(float(elements[5]))]
        top_left = [int(float(elements[6])), int(float(elements[7]))]

        pts = [top_left, bottom_left, bottom_right, top_right]
        position_labels.append(pts)
    print(word_labels, position_labels)
    # print(position_labels)

    one_image(img, position_labels)


# for i in range(10001, 10003):
#     test(i)
