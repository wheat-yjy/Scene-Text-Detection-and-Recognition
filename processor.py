# From Team Blue

import cv2
import time
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PSENet.utils.utils_tool import logger, cfg

from text_recognize.CRNN_FINAL import load_model as load_model_crnn
from text_recognize.CRNN_FINAL import one_image as recog_crnn

from PSENet.PSENET_TD import load_model as load_model_pse
from PSENet.PSENET_TD import predict as detect_pse

from CTPN.CTPN_TD import load_model as load_model_ctpn
from CTPN.CTPN_TD import predict as detect_ctpn



TEST_IMAGE_PATH = './img/'

R_IMAGE_PATH = './image/'

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


def init():
    load_model_ctpn()
    load_model_pse()
    load_model_crnn()


def resize_image(image, short_edge = 15):
    top, bottom, left, right = (0, 0, 0, 0)

    # 获取图片尺寸
    h, w, _ = image.shape
    height = h
    width = w
    # 计算短边需要增加多少像素宽度才能与长边等长(相当于padding，长边的padding为0，短边才会有padding)
    if h < short_edge:
        dh = short_edge - h
        top = dh // 2
        bottom = dh - top
        height = short_edge
        width = w
    elif w < short_edge:
        dw = short_edge - w
        left = dw // 2
        right = dw - left
        height = h
        width = short_edge
    else:
        pass  # pass是空语句，是为了保持程序结构的完整性。pass不做任何事情，一般用做占位语句。

    # RGB颜色
    BLACK = [0, 0, 0]
    # 给图片增加padding，使图片长、宽相等
    # top, bottom, left, right分别是各个边界的宽度，cv2.BORDER_CONSTANT是一种border type，表示用相同的颜色填充
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    # 调整图像大小并返回图像，目的是减少计算量和内存占用，提升训练速度
    return cv2.resize(constant, (width, height))

def get_rois(im, text_positions, rects):
    results = []
    cot = 0
    for point in text_positions:
        min_x = min(
            [int(float(point[0][0])), int(float(point[1][0])), int(float(point[2][0])), int(float(point[3][0]))])
        max_x = max(
            [int(float(point[0][0])), int(float(point[1][0])), int(float(point[2][0])), int(float(point[3][0]))])
        min_y = min(
            [int(float(point[0][1])), int(float(point[1][1])), int(float(point[2][1])), int(float(point[3][1]))])
        max_y = max(
            [int(float(point[0][1])), int(float(point[1][1])), int(float(point[2][1])), int(float(point[3][1]))])

        min_x = max(0,min_x-10)
        min_y = max(0,min_y-2)
        max_x = min(max_x+10, im.shape[1])
        max_y = min(max_y+2, im.shape[0])
        plt.imshow(im[min_y: max_y, min_x: max_x])
        plt.show()
        print(im.shape)
        plt.imshow(resize_image(im[min_y: max_y, min_x: max_x]))
        plt.show()
        print(im.shape)
        single_item = {}
        single_item['x0'] = min_x
        single_item['y0'] = min_y
        single_item['x1'] = max_x
        single_item['y1'] = max_y
        single_item['text'] = ''
        # 旋转
        # (h,w) = im.shape[:2]
        # center = (w//2, h//2)
        # M = cv2.getRotationMatrix2D(center, rects[cot][4], 1.0)
        # rotated = cv2.warpAffine(im, M, (w,h))
        # rotated = rotated[min_y: max_y, min_x: max_x]
        # plt.imshow(rotated)
        # plt.show()
        cv2.imwrite(R_IMAGE_PATH+'test.jpg',resize_image(im[min_y: max_y, min_x: max_x]) ,[int(cv2.IMWRITE_JPEG_QUALITY),100])
        print(single_item)
        ret = getTextFromPicture(R_IMAGE_PATH+'test.jpg')
        print(ret)
        if len(ret)!=0:
            single_item['text'] = ret[0]
        results.append(single_item)
    return results


def process(im, im_path="test.jpg"):
    """
    :param im: np.array
    :return: list of dict
    """
    boxes, rects = detect_pse(im, im_path)
    for i in boxes:
        print(i)
    print(type(rects))
    avg = 0
    bn = 0
    for i in rects:
        avg += i[4]
        if i[4] > -2:
            bn += 1
    avg /= len(rects)
    bn = bn * 1.0 / len(rects)
    print('avg: ', avg, 'bn: ', bn)
    if abs(avg) < 3 and bn > 0.8:
        logger.info('CTPN')
        boxes, rects = detect_ctpn(im, im_path)

    # plt.imshow(im)
    # plt.show()
    results = None

    results = recog_crnn(im, boxes)
    if rects is not None:
        results = info(results, rects)
    return results


def info(result, rects):
    for i in range(len(result)):
        result[i]['x0'] = int(rects[i, 0])
        result[i]['y0'] = int(rects[i, 1])
        result[i]['x1'] = int(rects[i, 2])
        result[i]['y1'] = int(rects[i, 3])
        result[i]['degree'] = float(rects[i, 4])
    # print(result)
    return result


def demo():
    init()
    im_names = get_images()
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        start = time.time()
        img = cv2.imread(im_name)
        process(img, im_name)
        duration = time.time() - start
        logger.info('[timing] {}'.format(duration))


if __name__ == '__main__':
    demo()
