# -*- coding:utf-8 -*-
import cv2
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import matplotlib.pyplot as plt

project_to_run = './PSENet/' # project path compare to run temrinal
tf.app.flags.DEFINE_string('test_data_path', project_to_run + 'tmp/images/', '')
tf.app.flags.DEFINE_string('gpu_list', '2', '')
tf.app.flags.DEFINE_string('checkpoint_path', project_to_run + 'resnet_train/', '')
tf.app.flags.DEFINE_string('output_dir', project_to_run + 'results/', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')

from PSENet.nets import model
from PSENet.pse import pse
from PSENet.utils.utils_tool import logger, cfg
FLAGS = tf.app.flags.FLAGS

logger.setLevel(cfg.debug)

TYPE_TALL_THRESHOLD = 1.01
TYPE_FAT_THRESHOLD = 0.99
COS_THRESHOLD = 0.7
SIM_X_INRECT_THRESHOLD = 50
SIM_Y_INRECT_THRESHOLD = 25
SIM_WIDTH_THRESHOLD = 5
MERGE_OFFSET_THRESHOLD = 15


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
    for label_value in label_values:
        # (y,x)
        points = np.argwhere(mask_res_resized == label_value)
        points = points[:, (1, 0)]
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
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

    boxes = merge(np.array(boxes))

    # boxes经过缩减，kernals不用管，不用修改
    return boxes, kernals, timer
    # return np.array(boxes), kernals, timer


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
    print(rec_type)
    print('x: ', point[0], '|', rect[0, 0], '|', rect[2, 0])
    print('y: ', point[1], '|', rect[0, 1], '|', rect[2, 1])
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
    print('boxes: ', boxes)
    ## TODO: Test
    # rect = cv2.minAreaRect(boxes)
    # box = cv2.boxPoints(rect)
    # box = boxes.tolist()
    # box = sorted(boxes)
    # temp = box[3]
    # box[3] = box[1]
    # box[1] = box[2]
    # box[2] = temp
    # ready_boxes = np.array(box)
    # _______________________________

    boxes = sorted(boxes)
    print(boxes)
    small_index = len(boxes) - 1
    big_index = 0
    for i in range(len(boxes)):
        if boxes[i][1] >= boxes[big_index][1]:
            big_index = i
        if boxes[len(boxes) - 1 - i][1] <= boxes[small_index][1]:
            small_index = len(boxes) - 1 - i

    print('small: ', boxes[small_index])
    print('big: ', boxes[big_index])
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
    print(ready_boxes)
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
    print(d)

    boxes_tem = np.zeros([boxes.shape[0] - len(d.items()) + len(set(d.values())), 4, 2])
    # print("boxes_tem: ", boxes_tem.shape)

    index = 0
    for i in range(len(boxes)):
        if i not in d:
            boxes_tem[index, :, :] = boxes[i, :, :]
            index += 1

    for i in set(d.values()):
        print(i, [j for j, p in d.items() if p == i])
        boxes_tem[index, :, :] = _merge(boxes[[j for j, p in d.items() if p == i], :, :])
        index += 1

    assert index == boxes_tem.shape[0], "ensure boxes_tem is full"

    # print(boxes_tem)
    return boxes_tem


def show_score_geo(color_im, kernels, im_res):
    fig = plt.figure()
    cmap = plt.cm.hot
    #
    ax = fig.add_subplot(241)
    im = kernels[0] * 255
    ax.imshow(im)

    ax = fig.add_subplot(242)
    im = kernels[1] * 255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(243)
    im = kernels[2] * 255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(244)
    im = kernels[3] * 255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(245)
    im = kernels[4] * 255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(246)
    im = kernels[5] * 255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(247)
    im = color_im
    ax.imshow(im)

    ax = fig.add_subplot(248)
    im = im_res
    ax.imshow(im)

    fig.show()


def predict(im):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        seg_maps_pred = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)

            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            logger.info('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            start_time = time.time()
            im_resized, (ratio_h, ratio_w) = resize_image(im)
            h, w, _ = im_resized.shape
            timer = {'net': 0, 'pse': 0}
            start = time.time()
            seg_maps = sess.run(seg_maps_pred, feed_dict={input_images: [im_resized]})
            timer['net'] = time.time() - start

            boxes, kernels, timer = detect(seg_maps=seg_maps, timer=timer, image_w=w, image_h=h)

            if boxes is not None:
                boxes = boxes.reshape((-1, 4, 2))
                boxes[:, :, 0] /= ratio_w
                boxes[:, :, 1] /= ratio_h
                h, w, _ = im.shape
                boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w)
                boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h)

            duration = time.time() - start_time
            logger.info('[timing] {}'.format(duration))

            # return boxes
            return im, boxes



def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    try:
        print(FLAGS.output_dir)
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        seg_maps_pred = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            print(FLAGS.checkpoint_path)
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)

            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            logger.info('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            for im_fn in im_fn_list:
                im = cv2.imread(im_fn)[:, :, ::-1]
                logger.debug('image file:{}'.format(im_fn))

                start_time = time.time()
                im_resized, (ratio_h, ratio_w) = resize_image(im)
                h, w, _ = im_resized.shape
                # options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
                # run_metadata = tf.RunMetadata()
                timer = {'net': 0, 'pse': 0}
                start = time.time()
                seg_maps = sess.run(seg_maps_pred, feed_dict={input_images: [im_resized]})
                timer['net'] = time.time() - start
                # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                # chrome_trace = fetched_timeline.generate_chrome_trace_format()
                # with open(os.path.join(FLAGS.output_dir, os.path.basename(im_fn).split('.')[0]+'.json'), 'w') as f:
                #     f.write(chrome_trace)

                boxes, kernels, timer = detect(seg_maps=seg_maps, timer=timer, image_w=w, image_h=h)
                logger.info('{} : net {:.0f}ms, pse {:.0f}ms'.format(
                    im_fn, timer['net'] * 1000, timer['pse'] * 1000))

                if boxes is not None:
                    boxes = boxes.reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h
                    h, w, _ = im.shape
                    boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w)
                    boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h)

                duration = time.time() - start_time
                logger.info('[timing] {}'.format(duration))

                # save to file
                if boxes is not None:
                    res_file = os.path.join(
                        FLAGS.output_dir,
                        '{}.txt'.format(os.path.splitext(
                            os.path.basename(im_fn))[0]))

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
                            cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True,
                                          color=(255, 255, 0), thickness=2)
                if not FLAGS.no_write_images:
                    img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
                    cv2.imwrite(img_path, im[:, :, ::-1])
                # show_score_geo(im_resized, kernels, im)


if __name__ == '__main__':
    tf.app.run()
