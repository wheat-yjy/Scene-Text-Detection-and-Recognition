# -*- coding:utf-8 -*-
import cv2
import time
import os
import numpy as np
import math
import random
import tensorflow as tf
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
from text_recognize.CRNN_FT import one_image
from text_recognize.CRNN_FT import load_model

