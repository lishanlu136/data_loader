#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2019/1/24 9:59
@Author     : Li Shanlu
@File       : data_reader_main.py
@Software   : PyCharm
@Description: 利用dataset读取图片数据
"""
import tensorflow as tf
import os
import sys
import numpy as np
import cv2
from data_loader import DataLoader
from matplotlib import pyplot as plt


def get_img_path_and_lab(data_dir):
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for img_name in os.listdir(data_dir):
        img_path = data_dir + '/' + img_name
        if 'cat' in img_name:
            cats.append(img_path)
            label_cats.append(0)
        else:
            dogs.append(img_path)
            label_dogs.append(1)
    print("There are %d cats\nThere are %d dogs" % (len(cats), len(dogs)))
    return cats, label_cats, dogs, label_dogs


def main():
    data_dir = "dataset/train"
    cats, label_cats, dogs, label_dogs = get_img_path_and_lab(data_dir)
    train_img_list = cats + dogs
    train_label_list = label_cats + label_dogs
    dataset = DataLoader(train_img_list, train_label_list, [160, 160])
    next_element, init_op = dataset.data_batch(augment=True, shuffle=True, batch_size=5, repeat_times=2, num_threads=4, buffer=30)
    sess = tf.Session()
    with sess.as_default():
        sess.run(init_op)
        i = 0
        try:
            while True:
                i += 1
                next_element_data = sess.run(next_element)
                image_data = next_element_data[0]
                label = next_element_data[1]
                for j in range(5):                                   # for batchsize
                    print(label[j])
                    img = np.asarray(image_data[j,:,:,:], dtype='uint8')
                    #mean = np.mean(img)
                    #std = np.std(img)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img_name = 'read/test_%d_%d_%d.jpg' % (i, j, label[j])
                    cv2.imwrite(img_name, img)
                    plt.imshow(img)
                    plt.show()
        except tf.errors.OutOfRangeError:
            print("end.")


if __name__ == '__main__':
    main()