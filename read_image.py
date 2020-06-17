#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 20-3-27 上午10:26
@Author     : lishanlu
@File       : read_image.py
@Software   : PyCharm
@Description:
"""

from __future__ import division, print_function, absolute_import
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

JPG_image = './dataset/train/cat.1.jpg'
PNG_image = './0_0.png'


def read_image_from_cv2(filename):
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


image_content = tf.read_file(PNG_image)
#image = tf.image.decode_image(image_content, channels=3)
#print(image)                                                  # Tensor("decode_image/cond_jpeg/Merge:0", dtype=uint8)
image = tf.image.decode_png(image_content, channels=3)
print(image)                                                   # Tensor("DecodePng:0", shape=(?, ?, 3), dtype=uint8)
#image = tf.py_func(read_image_from_cv2, [PNG_image], tf.uint8)
#print(image)                                                   # Tensor("PyFunc:0", dtype=uint8)
#image = tf.random_crop(image, [112,112,3])
#print(image)                                                   # Tensor("random_crop:0", shape=(112, 112, 3), dtype=uint8)
#image.set_shape([None, None, 3])
#print(image)                                                    # Tensor("PyFunc:0", shape=(?, ?, 3), dtype=uint8)
image = tf.image.resize_images(image, [112, 112])
print(image)                                                    # Tensor("Squeeze:0", shape=(112, 112, 3), dtype=uint8)
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
print(image)                                                    # Tensor("Identity:0", shape=(112, 112, 3), dtype=float32)
#image = tf.image.per_image_standardization(image)
#print(image)                                                   # Tensor("div:0", shape=(112, 112, 3), dtype=float32)
with tf.Session() as sess:
    img_rgb = sess.run(image)                    # RGB通道
    print(img_rgb.shape)
    #img_rgb = np.asarray(img_rgb[:, :, :], dtype='uint8')
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    #cv2.imshow('cv2_cat', img_bgr)
    #cv2.waitKey()

    plt.imshow(img_rgb)
    plt.show()

