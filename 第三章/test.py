#!/usr/bin/env python
# encoding: utf-8

"""
@Author : wangzhaoyun
@Contact:1205108909@qq.com
@File : test.py 
@Time : 2020/7/21 9:08 
"""
# import tensorflow as tf
# # tf.test.is_gpu_available('GPU')
# tf.config.list_physical_devices('GPU')
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255)
