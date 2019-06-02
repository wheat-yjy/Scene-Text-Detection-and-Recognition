import tensorflow as tf
import keras
from keras import backend as K
K.set_image_dim_ordering('tf')
import itertools
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Deconvolution2D
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Softmax
from keras.layers import Reshape, Lambda
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD

import os
import numpy as np
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import pickle
file = open('text_recognize/label_map_ft.pickle', 'rb+')
label_map = pickle.load(file)

TEST_FUNC = None


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, :, :]
    input_length = keras.backend.reshape(input_length, (1, 1))
    label_length = keras.backend.reshape(label_length, (1, 1))
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
        if c == label_map['#']:  # CTC Blank
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


def concat(args):
    a, b = args
    return tf.concat((a, b), axis=2)


def load_model():
    input_data = Input(name='the_input', shape=[100, 32, 1], dtype='float32', batch_shape=[1, 100, 32, 1])

    inner = Conv2D(64, [3, 3], padding='same', activation='relu', kernel_initializer='he_normal', name='conv1')(
        input_data)
    print(inner.shape, "conv1")
    inner = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', name='max_pooling1')(inner)
    print(inner.shape, "max1")

    inner = Conv2D(128, [3, 3], padding='same', activation='relu', kernel_initializer='he_normal', name='conv2')(inner)
    print(inner.shape, "conv2")
    inner = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', name='max_pooling2')(inner)
    print(inner.shape, "max2")

    inner = Conv2D(256, [3, 3], padding='same', activation='relu', kernel_initializer='he_normal', name='conv3')(inner)
    print(inner.shape, "conv3")
    inner = Conv2D(256, [3, 3], padding='same', activation='relu', kernel_initializer='he_normal', name='conv4')(inner)
    print(inner.shape, "conv4")

    inner = MaxPooling2D(pool_size=(1, 1), strides=[1, 2], name='max_pooling3', padding='same')(inner)
    print(inner.shape, "max3")
    inner = Conv2D(512, [3, 3], padding='same', activation='relu', kernel_initializer='he_normal', name='conv5')(inner)
    print(inner.shape, "conv5")
    inner = BatchNormalization()(inner)
    print(inner.shape, "bn1")

    inner = Conv2D(512, [3, 3], padding='same', activation='relu', kernel_initializer='he_normal', name='conv6')(inner)
    print(inner.shape, "conv6")
    inner = BatchNormalization()(inner)
    print(inner.shape, "bn2")

    inner = MaxPooling2D(pool_size=(1, 1), strides=[1, 2], padding='valid', name='max_pooling4')(inner)
    print(inner.shape, "max4")
    inner = Conv2D(512, [2, 2], padding='valid', activation='relu', kernel_initializer='he_normal', name='conv7')(inner)
    print(inner.shape, "conv7")

    inner = Reshape((24, 512))(inner)

    gru = GRU(256, return_sequences=True,
              kernel_initializer='he_normal', name='gru')(inner)
    gru_b = GRU(256, return_sequences=True,
                go_backwards=True, kernel_initializer='he_normal',
                name='gru_b')(inner)
    gru_merged = Lambda(concat)([gru, gru_b])

    print(gru_merged.shape, "gru_merged")

    y_pred = Dense(len(label_map) + 3, activation='softmax', kernel_initializer='he_normal', name='dense_out')(
        gru_merged)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels',
                   shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=1e-4, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_data, labels, input_length, label_length],
                  outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    weight_file = 'text_recognize/weights%02d.h5' % (400)
    model.load_weights(weight_file)
    # captures output of softmax so we can decode the output during visualization
    test_func = K.function([input_data], [y_pred])
    global TEST_FUNC
    if TEST_FUNC is None:
        TEST_FUNC = test_func

def one_image(image, text_positions):
    results = []
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for point in text_positions:
        min_x = min([int(float(point[0][0])), int(float(point[1][0])), int(float(point[2][0])), int(float(point[3][0]))])
        max_x = max([int(float(point[0][0])), int(float(point[1][0])), int(float(point[2][0])), int(float(point[3][0]))])
        min_y = min([int(float(point[0][1])), int(float(point[1][1])), int(float(point[2][1])), int(float(point[3][1]))])
        max_y = max([int(float(point[0][1])), int(float(point[1][1])), int(float(point[2][1])), int(float(point[3][1]))])
        single_regular_area = image_gray[min_y:max_y, min_x:max_x] / 256
        # cv2.imwrite("test.jpg", single_regular_area)
        single_regular_area = cv2.resize(single_regular_area, (100, 32), interpolation=cv2.INTER_CUBIC)

        # print("看这里这里有坐标\t {} \t".format(points)) # 我弄的。。
        X_data = np.reshape(single_regular_area, [1, 100, 32, 1])

        global TEST_FUNC
        text = TEST_FUNC([X_data])[0]
        # print(text)
        ret = []

        for j in range(text.shape[0]):
            out_best = list(np.argmax(text[j, 0:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            outstr = labels_to_text(out_best)
            ret.append(outstr)
        # print("识别结果为:{}".format(ret))

        single_item = {}
        single_item['x0'] = min_x
        single_item['y0'] = min_y
        single_item['x1'] = max_x
        single_item['y1'] = max_y
        single_item['text'] = ret[0]
        if '滚出西藏' in single_item['text']:
            single_item['tag'] = 'red'
        else:
            single_item['tag'] = ''
        results.append(single_item)

        # cv2.imwrite("test.jpg", single_regular_area)
        # regular_text_images.append(single_regular_area)
    print(results)
    return results

import cv2
def test(index):
    image_file_name = "/home/benke/veteranjy/data/images/tr_img_" + str(index) + ".jpg"
    # image_file_name = "/home/benke/xuyichu/PSENet/tmp/tr_img_03001.jpg"
    img = cv2.imread(image_file_name, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

    one_image(image, position_labels)

# load_model()
# for i in range(10001, 10005):
#     test(i)
