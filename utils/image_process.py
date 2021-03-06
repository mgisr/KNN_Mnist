#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/21 16:50
# @Author  : mgisr
# @Site    :
# @File    : image_process.py
# @Software: PyCharm

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def normal(data, val):
    """
    对图片进行归一化
    :param data: 将要归一化的值
    :param val: 简单除法归一化的除数
    :return: 归一化后的数据
    """

    data = np.array(data)
    return data / val


def rgb2gray(img):
    """
    RGB图像转灰度图
    :param img: RGB图像
    :return: 转换后的灰度图
    """

    return Image.fromarray(img, mode='RGB').convert('L')


def callout(img, label):
    """
    将检测出的结果标注到图片上
    :param img:
    :param label:
    :return: 标注后的图片
    """

    t = np.array(img)
    plt.imshow(t)
    plt.text(5, 5, str(label), fontsize=60, color='r')

    return t
