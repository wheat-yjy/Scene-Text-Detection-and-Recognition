from __future__ import print_function

import glob
import os
import shutil
import sys
import time
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

DEBUG_MODE = False

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

sys.path.append(os.getcwd())
from CTPN.lib.fast_rcnn.config import cfg, cfg_from_file
from CTPN.lib.fast_rcnn.test import _get_blobs
from CTPN.lib.text_connector.detectors import TextDetector
from CTPN.lib.text_connector.text_connect_cfg import Config as TextLineCfg
from CTPN.lib.rpn_msr.proposal_layer_tf import proposal_layer

DEBUG_MODE = False
PROJECT_COMPARE_PATH = './CTPN/' # project path compare to run temrinal
TEST_IMAGE_PATH = './img/'
TEST_OUTPUT_PATH = './ctpn_output/' # 调试用的图片和文本框输出路径

NET_CONFIG_PATH = PROJECT_COMPARE_PATH + 'ctpn/text.yml'
NET_MODEL_PATH = PROJECT_COMPARE_PATH + 'data/ctpn.pb'

config = None
sess = None
input_img = None
output_cls_prob = None
output_box_pred = None

def load_model():
    global config
    global sess
    global input_img
    global output_cls_prob
    global output_box_pred

    cfg_from_file(NET_CONFIG_PATH)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # init session
    with gfile.FastGFile(NET_MODEL_PATH, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    sess.run(tf.global_variables_initializer())

    input_img = sess.graph.get_tensor_by_name('Placeholder:0')
    output_cls_prob = sess.graph.get_tensor_by_name('Reshape_2:0')
    output_box_pred = sess.graph.get_tensor_by_name('rpn_bbox_pred/Reshape_1:0')


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def draw_boxes(img, boxes, scale, image_name):
    base_name = image_name.split('/')[-1]
    with open(TEST_OUTPUT_PATH + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

            min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))

            line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\r\n'
            f.write(line)

    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(TEST_OUTPUT_PATH, base_name), img)

def produce_normal_boxes(boxes,scale):
    ret_boxes = []
    for box in boxes:
        min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
        min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
        max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
        max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
        box_ = [[min_x,min_y],[max_x,min_y],[max_x,max_y],[min_x,max_y]]
        ret_boxes.append(box_)

    return ret_boxes

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(TEST_IMAGE_PATH):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    logger.info('Find {} images'.format(len(files)))
    return files

def predict(img, im_name="test.jpg", debug=False):
    global DEBUG_MODE
    DEBUG_MODE = debug

    if DEBUG_MODE:
        if os.path.exists(TEST_OUTPUT_PATH):
            shutil.rmtree(TEST_OUTPUT_PATH)
        os.makedirs(TEST_OUTPUT_PATH)

    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    blobs, im_scales = _get_blobs(img, None)
    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)
    cls_prob, box_pred = sess.run([output_cls_prob, output_box_pred], feed_dict={input_img: blobs['data']})
    rois, _ = proposal_layer(cls_prob, box_pred, blobs['im_info'], 'TEST', anchor_scales=cfg.ANCHOR_SCALES)

    scores = rois[:, 0]
    boxes = rois[:, 1:5] / im_scales[0]
    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    if DEBUG_MODE:
        draw_boxes(img, boxes, scale, im_name)
    # up box (-1,4)
    ret_boxes = produce_normal_boxes(boxes, scale)
    # (-1,4,2) list格式

    return ret_boxes, None



# if __name__ == '__main__':
#     load_model()
#     im_names = get_images()
#
#     for im_name in im_names:
#         print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#         print(('Demo for {:s}'.format(im_name)))
#         start = time.time()
#         img = cv2.imread(im_name)
#         predict(img)
#         print(time.time() - start)