# -*- coding:utf-8 -*-
import cv2
import time
import os
import shutil
import numpy as np
import math
import random
import tensorflow as tf
from PSENet.utils.utils_tool import logger, cfg
import matplotlib.pyplot as plt

from PSENet.nets import model
from PSENet.pse import pse

project_to_run = './PSENet/'  # project path compare to run temrinal
TEST_IMAGE_PATH = './img/'
TEST_OUTPUT_PATH = './pse_output/'  # 调试用的图片和文本框输出路径
DEBUG_MODE = False

tf.app.flags.DEFINE_string('test_data_path', TEST_IMAGE_PATH, '')
tf.app.flags.DEFINE_string('gpu_list', '2', '')
tf.app.flags.DEFINE_string('checkpoint_path', project_to_run + 'resnet_train/', '')
tf.app.flags.DEFINE_string('output_dir', TEST_OUTPUT_PATH, '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')

FLAGS = tf.app.flags.FLAGS

logger.setLevel(cfg.debug)

TYPE_TALL_THRESHOLD = 1.01
TYPE_FAT_THRESHOLD = 0.99
COS_THRESHOLD = 0.7
SIM_X_INRECT_THRESHOLD = 50
SIM_Y_INRECT_THRESHOLD = 25
SIM_WIDTH_THRESHOLD = 5
MERGE_OFFSET_THRESHOLD = 15

# print(FLAGS.gpu_list)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

graph = tf.get_default_graph()

session = None

input_images = None  # 输入
global_step = None
seg_maps_pred = None  # 输出


def load_model():
    global session
    global input_images
    global global_step
    global seg_maps_pred

    with graph.as_default():
        session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        seg_maps_pred = model.model(input_images, is_training=False)
        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        print(FLAGS.checkpoint_path)
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
        model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
        logger.info('Restore from {}'.format(model_path))
        saver.restore(session, model_path)


def predict(im, im_path='test.jpg', debug=False):
    global DEBUG_MODE
    DEBUG_MODE = debug
    if DEBUG_MODE:
        if os.path.exists(TEST_OUTPUT_PATH):
            shutil.rmtree(TEST_OUTPUT_PATH)
        os.makedirs(TEST_OUTPUT_PATH)
    im = im
    im_resized, (ratio_h, ratio_w) = resize_image(im)
    h, w, _ = im_resized.shape
    timer = {'net': 0, 'pse': 0}
    start = time.time()
    with graph.as_default():
        seg_maps = session.run(seg_maps_pred, feed_dict={input_images: [im_resized]})
        timer['net'] = time.time() - start

        boxes, kernels, timer, rects = detect(seg_maps=seg_maps, timer=timer, image_w=w, image_h=h)

        if boxes is not None:
            boxes = boxes.reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h
            h, w, _ = im.shape
            boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w)
            boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h)

        if rects is not None:
            rects = np.array(rects).reshape((-1, 5))
            rects[:, [0, 2]] /= ratio_w  # x
            rects[:, [1, 3]] /= ratio_h  # y
            h, w, _ = im.shape
            rects[:, [0, 2]] = np.clip(rects[:, [0, 2]], 0, w)
            rects[:, [1, 3]] = np.clip(rects[:, [1, 3]], 0, h)

        if DEBUG_MODE:
            if boxes is not None:
                res_file = os.path.join(
                    TEST_OUTPUT_PATH,
                    '{}.txt'.format(os.path.splitext(
                        os.path.basename(im_path))[0]))

                with open(res_file, 'w') as f:
                    num = 0
                    for i in range(len(boxes)):
                        # to avoid submitting errors
                        box = boxes[i]
                        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                            continue

                        num += 1

                        f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                            box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1]))
                        cv2.polylines(im, [box.astype(np.int32).reshape((-1, 1, 2))], True,
                                      color=(255, 255, 0), thickness=2)
                img_path = os.path.join(TEST_OUTPUT_PATH, os.path.basename(im_path))
                cv2.imwrite(img_path, im)

        boxes = boxes.tolist()
        return boxes, rects


def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    logger.info('Find {} images'.format(len(files)))
    return files


def resize_image(im, max_side_len=1200):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.

    # ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w

    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 + 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 + 1) * 32
    logger.info('resize_w:{}, resize_h:{}'.format(resize_w, resize_h))
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(seg_maps, timer, image_w, image_h, min_area_thresh=10, seg_map_thresh=0.9, ratio=1):
    '''
    restore text boxes from score map and geo map
    :param seg_maps:
    :param timer:
    :param min_area_thresh:
    :param seg_map_thresh: threshhold for seg map
    :param ratio: compute each seg map thresh
    :return:
    '''
    if len(seg_maps.shape) == 4:
        seg_maps = seg_maps[0, :, :, ]
    # get kernals, sequence: 0->n, max -> min
    kernals = []
    one = np.ones_like(seg_maps[..., 0], dtype=np.uint8)
    zero = np.zeros_like(seg_maps[..., 0], dtype=np.uint8)
    thresh = seg_map_thresh
    for i in range(seg_maps.shape[-1] - 1, -1, -1):
        kernal = np.where(seg_maps[..., i] > thresh, one, zero)
        kernals.append(kernal)
        thresh = seg_map_thresh * ratio
    start = time.time()
    mask_res, label_values = pse(kernals, min_area_thresh)
    timer['pse'] = time.time() - start
    mask_res = np.array(mask_res)
    mask_res_resized = cv2.resize(mask_res, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
    boxes = []
    rects = []
    for label_value in label_values:
        # (y,x)
        points = np.argwhere(mask_res_resized == label_value)
        points = points[:, (1, 0)]
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        # print('no fix: ', rect)

        rects.append([rect[0][0] - rect[1][0] / 2, rect[0][1] - rect[1][1] / 2,
                      rect[0][0] + rect[1][0] / 2, rect[0][1] + rect[1][1] / 2,
                      rect[2]])

        sort_box = box[np.argsort(box[:, -1])[::-1]]  # 按Y坐标从大到小排序，改变一些框的旋转方向
        if sort_box[0, 1] > sort_box[1, 1]:
            if sort_box[0, 0] > rect[0][0]:
                rects[-1][4] = 90 + rects[-1][4]

        if rects[-1][4]>0 or rect[2] < -85:
            rects[-1] = [rect[0][0] - rect[1][1] / 2, rect[0][1] - rect[1][0] / 2,
                         rect[0][0] + rect[1][1] / 2, rect[0][1] + rect[1][0] / 2,
                         90 + rect[2]]


        # print('fix: ', rects[-1])
        # map(int, rect)
        # print("box(no sorted): ", box)
        box = box.tolist()
        box = sorted(box)
        temp = box[3]
        box[3] = box[1]
        box[1] = box[2]
        box[2] = temp
        box = np.array(box)
        # print("box(sorted): ", box)
        boxes.append(box)
    # boxes = merge(np.array(boxes))

    # boxes经过缩减，kernals不用管，不用修改
    return np.array(boxes), kernals, timer, rects


def new_Locate(boxes):
    """

    :param boxes:
    :return:
    """
    boxes_new = np.zeros((boxes.shape[0], 4), dtype=np.float)
    degrees = np.zeros((boxes.shape[0], 1))
    unit = np.array([0, 1])
    print('test', boxes)
    for i in range(len(boxes)):
        box = boxes[i]
        # (box)
        low_ctr = (box[0] + box[3]) / 2  # 左上为0，右下为2
        up_ctr = (box[1] + box[2]) / 2
        left_ctr = (box[0] + box[1]) / 2
        right_ctr = (box[2] + box[3]) / 2
        center = (low_ctr + up_ctr) / 2
        print(box)
        x_len = np.linalg.norm(right_ctr - left_ctr)
        y_len = np.linalg.norm(up_ctr - low_ctr)
        # print(x_len, y_len)
        boxes_new[i, 0] = center[0] - x_len / 2
        boxes_new[i, 1] = center[1] - y_len / 2
        boxes_new[i, 2] = center[0] + x_len / 2
        boxes_new[i, 3] = center[1] + y_len / 2
        temp = compute_direct(right_ctr - left_ctr, unit)
        if left_ctr[0] < right_ctr[0]:
            degrees[i] = temp
        else:
            degrees[i] = (-1) * temp
    return boxes_new, degrees


# 弃用
def judgeType(boxes):
    rec_type = np.zeros((len(boxes),))
    boxes_side_ctr = np.zeros_like(boxes, dtype=np.float)
    for i in range(len(boxes)):
        box = boxes[i]
        # print(box)
        low_ctr = (box[0] + box[1]) / 2
        up_ctr = (box[2] + box[3]) / 2
        left_ctr = (box[0] + box[3]) / 2
        right_ctr = (box[1] + box[2]) / 2

        boxes_side_ctr[i, 0, :] = low_ctr
        boxes_side_ctr[i, 1, :] = up_ctr
        boxes_side_ctr[i, 2, :] = left_ctr
        boxes_side_ctr[i, 3, :] = right_ctr

        # print(low_ctr, up_ctr, left_ctr, right_ctr)
        x_len = np.linalg.norm(right_ctr - left_ctr)
        y_len = np.linalg.norm(up_ctr - low_ctr)
        ratio = y_len / x_len
        # print('y_len: ' , y_len, 'x_len: ',x_len, 'ratio: ', ratio)
        if ratio <= TYPE_FAT_THRESHOLD:
            rec_type[i] = 0
        elif ratio > TYPE_FAT_THRESHOLD and ratio < TYPE_TALL_THRESHOLD:
            rec_type[i] = 1
        elif ratio >= TYPE_TALL_THRESHOLD:
            rec_type[i] = 2

    print(rec_type)
    return boxes_side_ctr, rec_type


def isPointInRect(point, rect, rec_type):
    # print(rec_type)
    # print('x: ', point[0], '|', rect[0, 0], '|', rect[2, 0])
    # print('y: ', point[1], '|', rect[0, 1], '|', rect[2, 1])
    if rec_type == 0:
        if point[0] < rect[0, 0] - SIM_X_INRECT_THRESHOLD or point[0] > rect[2, 0] + SIM_X_INRECT_THRESHOLD:
            return False
        if point[1] < rect[0, 1] - SIM_Y_INRECT_THRESHOLD or point[1] > rect[2, 1] + SIM_Y_INRECT_THRESHOLD:
            return False
    elif rec_type == 2:
        if point[0] < rect[0, 0] - SIM_Y_INRECT_THRESHOLD or point[0] > rect[2, 0] + SIM_Y_INRECT_THRESHOLD:
            return False
        if point[1] < rect[0, 1] - SIM_X_INRECT_THRESHOLD or point[1] > rect[2, 1] + SIM_X_INRECT_THRESHOLD:
            return False
    return True


def _merge(boxes):
    boxes = boxes.reshape([-1, 2])
    boxes = boxes.tolist()
    # print('boxes: ', boxes)
    boxes = sorted(boxes)
    # print(boxes)
    small_index = len(boxes) - 1
    big_index = 0
    for i in range(len(boxes)):
        if boxes[i][1] >= boxes[big_index][1]:
            big_index = i
        if boxes[len(boxes) - 1 - i][1] <= boxes[small_index][1]:
            small_index = len(boxes) - 1 - i

    # print('small: ', boxes[small_index])
    # print('big: ', boxes[big_index])
    ready_boxes = np.zeros([4, 2])
    if (abs(boxes[0][0] - boxes[small_index][0]) <= MERGE_OFFSET_THRESHOLD):
        ready_boxes[0, :] = boxes[small_index]
    else:
        ready_boxes[0, :] = boxes[0]

    if (abs(boxes[-1][0] - boxes[big_index][0]) <= MERGE_OFFSET_THRESHOLD):
        ready_boxes[2, :] = boxes[big_index]
    else:
        ready_boxes[2, :] = boxes[-1]

    ready_boxes[1, :] = [ready_boxes[2][0], ready_boxes[0][1]]
    ready_boxes[3, :] = [ready_boxes[0][0], ready_boxes[2][1]]
    # print(ready_boxes)
    return ready_boxes


def merge(boxes):
    print("---- info ----")
    print("boxes shape: ", np.array(boxes).shape)
    print("Type")
    boxes_side_ctr, rec_type = judgeType(boxes)
    # 1:2-n 2:3-n 3:4-n
    # 层次遍历

    match_list = []
    # 1. 先检查非常近的/重合的框，然后合并
    for i in range(len(rec_type)):
        for j in range(i + 1, len(rec_type)):
            if np.abs(rec_type[i] - rec_type[j]) < 2:
                if rec_type[i] == 0 or rec_type[j] == 0:
                    # 宽,计算两个
                    # 计算出左右两点的向量
                    vector_a = boxes_side_ctr[i, 2, :] - boxes_side_ctr[i, 3, :]
                    vector_b = boxes_side_ctr[j, 2, :] - boxes_side_ctr[j, 3, :]
                    width_a = np.linalg.norm(boxes_side_ctr[i, 0, :] - boxes_side_ctr[i, 1, :])
                    width_b = np.linalg.norm(boxes_side_ctr[j, 0, :] - boxes_side_ctr[j, 1, :])
                    # 比较余弦相似度
                    if np.dot(vector_a, vector_b) / (
                            np.linalg.norm(vector_a) * np.linalg.norm(vector_b)) >= COS_THRESHOLD:
                        if isPointInRect(boxes_side_ctr[j, 2, :], boxes[i], 0) or isPointInRect(boxes_side_ctr[j, 3, :],
                                                                                                boxes[i], 0):
                            if np.abs(width_a - width_b) < SIM_WIDTH_THRESHOLD:
                                if boxes[i, 0, 1] < boxes[j, 0, 1]:
                                    match_list.append([i, j])
                                else:
                                    match_list.append([j, i])
                elif rec_type[i] == 2 or rec_type[j] == 2:
                    # 高,计算两个
                    # 计算出上下两点的向量
                    vector_a = boxes_side_ctr[i, 0, :] - boxes_side_ctr[i, 1, :]
                    vector_b = boxes_side_ctr[j, 0, :] - boxes_side_ctr[j, 1, :]
                    width_a = np.linalg.norm(boxes_side_ctr[i, 2, :] - boxes_side_ctr[i, 3, :])
                    width_b = np.linalg.norm(boxes_side_ctr[j, 2, :] - boxes_side_ctr[j, 3, :])
                    # 比较余弦相似度
                    if np.dot(vector_a, vector_b) / (
                            np.linalg.norm(vector_a) + np.linalg.norm(vector_b)) >= COS_THRESHOLD:
                        if isPointInRect(boxes_side_ctr[j, 0, :], boxes[i], 2) or isPointInRect(boxes_side_ctr[j, 1, :],
                                                                                                boxes[i], 2):
                            if np.abs(width_a - width_b) < SIM_WIDTH_THRESHOLD:
                                if boxes[i, 0, 1] < boxes[j, 0, 1]:
                                    match_list.append([i, j])
                                else:
                                    match_list.append([j, i])

    print("matchlist: ", match_list)

    # 2. 给每个框分类吧
    d = {}
    for i, j in match_list:
        if i not in d:
            d[i] = i
        if j not in d:
            d[j] = j
        d[d[j]] = d[i]
    # print(d)

    boxes_tem = np.zeros([boxes.shape[0] - len(d.items()) + len(set(d.values())), 4, 2])
    # print("boxes_tem: ", boxes_tem.shape)

    index = 0
    for i in range(len(boxes)):
        if i not in d:
            boxes_tem[index, :, :] = boxes[i, :, :]
            index += 1

    for i in set(d.values()):
        # print(i, [j for j, p in d.items() if p == i])
        boxes_tem[index, :, :] = _merge(boxes[[j for j, p in d.items() if p == i], :, :])
        index += 1

    assert index == boxes_tem.shape[0], "ensure boxes_tem is full"

    # print(boxes_tem)
    return boxes_tem


def fix(boxes):
    """
    返回的是一个正框，加上旋转（x1,x1,x3,x3） 计算出斜率
    :param boxes: (?,4,2) (左上开始，逆时针)
    :return: degrees: (?,1)
    """
    # method1: 两条对角线的交点，到右边两个点的向量。
    degrees = np.zeros((boxes.shape[0], 1))
    unit = np.array([0, 1])
    for i in range(boxes.shape[0]):
        degrees[i] = compute_direct(boxes[i, 2, :] - boxes[i, 0, :] + boxes[i, 3, :] - boxes[i, 1, :], unit)

    return degrees


def comput_Num(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_direct(a, b):
    """
    compute degree between vector a and vector b
    :param a: np.array (2,1)
    :param b: np.array (2,1)
    :return: degree
    """
    return math.acos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) / math.pi * 180

# if __name__ == '__main__':
#     load_model()
#     im_fn = get_images()
#     for i in im_fn:
#         im = cv2.imread(i)
#         predict(im)
