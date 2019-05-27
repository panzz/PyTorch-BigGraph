#!/usr/bin/env python
#-*- coding:utf-8 -*-


'''
  接k_means.py
  k_means.py中得到三维规范化数据data_zs；
  r增加了最后一列，列索引为“聚类类别”
'''
from __future__ import print_function, division, absolute_import
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import numpy
from numpy import *
import numpy as np
import random
from random import choice
from array import array

from scipy.spatial.distance import squareform, pdist
import sklearn
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities, _kl_divergence)
# from sklearn.utils.extmath import _ravel

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

import seaborn as sns
import os

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

examples_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
fileName = os.path.basename(os.path.realpath(__file__))
# print(examples_folder, fileName)

fdata_test = examples_folder + '/data/fb15k/fb15k_data_test.tsv'
ftarget_test = examples_folder + '/data/fb15k/fb15k_label_test.tsv'

fdata = examples_folder + '/data/fb15k/fb15k_data.tsv'  # "3.txt"
ftarget = examples_folder + '/data/fb15k/fb15k_label.tsv'  # "4.txt"


class chj_data_test(object):
    def __init__(self, data, target):
        self.data = data
        self.target = target


def chj_load_file_test(fdata, ftarget):
    data = numpy.loadtxt(fdata, dtype=float32)
    target = numpy.loadtxt(ftarget, dtype='str')

    print(data.shape)
    print(target.shape)
    # pexit()

    res = chj_data_test(data, target)
    return res


class chj_data(object):
    def __init__(self, data, target):
        self.data = data
        self.target = target


def chj_load_file(fdata, ftarget):
    data = numpy.loadtxt(fdata, delimiter='\t', dtype=float32)
    target = numpy.loadtxt(ftarget, dtype='str')  # int32)

    print(data.shape)
    print(target.shape)
    # pexit()

    res = chj_data(data, target)
    return res


def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    # for i in range(10):
    #     # Position of each label.
    #     xtext, ytext = np.median(x[colors == i, :], axis=0)
    #     txt = ax.text(xtext, ytext, str(i), fontsize=24)
    #     txt.set_path_effects([
    #         PathEffects.Stroke(linewidth=5, foreground="w"),
    #         PathEffects.Normal()])
    #     txts.append(txt)
    
    return f, ax, sc, txts


RS = 20150101


def run_iris():
  # iris = load_iris() # 使用sklearn自带的测试文件
  # iris = chj_load_file_test(fdata_test, ftarget_test)
  iris = chj_load_file(fdata, ftarget)

  '''
    n_components:   (default: 2) 嵌入空间的尺寸。
    learning_rate:  (default: 200.0) 
                    t-SNE的学习率通常在[10.0,1000.0]的范围内。如果
                    学习率太高，数据可能看起来像任何“球”
                    点与其最近邻居大致等距。如果
                    学习率太低，大多数点可能看起来压缩密集
                    云与少数异常值。如果成本函数陷入困境中
                    最低提高学习率可能会有所帮助。
    perplexity :    (default: 30)
                    困惑与其他流形学习算法中使用的最近邻居的数量有关。 
                    较大的数据集通常需要更大的困惑。 
                    考虑选择5到50之间的值。不同的值可能会导致显着不同的结果。
    random_state:   如果是int，则random_state是随机数生成器使用的种子;如果
                    RandomState实例，random_state是随机数生成器;如果没有，
                    随机数生成器是np.random使用的RandomState实例。
                    请注意，不同的初始化可能会导致不同的局部最小值
                    成本函数。
  '''
  X_tsne = TSNE(n_components=2, learning_rate=100).fit_transform(iris.data)
  # X_tsne = TSNE(random_state=RS).fit_transform(iris.data)
  # xtslist = list(X_tsne)
  # print("xtslist:{}, {}".format(xtslist, X_tsne.tolist))
  # X_pca = PCA().fit_transform(iris.data)
  print("finishe X_tsne!")
  plt.figure(figsize=(12, 6))
  point_numbers = list(range(len(iris.target)))
  plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=point_numbers)
  for i in range(X_tsne.shape[0]):
    plt.text(X_tsne[i, 0], X_tsne[i, 1], str(iris.target[i]), color='blue', fontdict={'weight': 'bold', 'size': 9})
  plt.colorbar()
  plt.savefig('fb15k_tsne-generated.png', format='png', transparent=True, dpi=300, pad_inches = 0)
  plt.show()

  # X_tsne = TSNE(learning_rate=100).fit_transform(iris.data)
  # X_pca = PCA().fit_transform(iris.data)
  # print("finishe!")
  # plt.figure(figsize=(10, 5))
  # plt.subplot(121)
  # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=iris.target)
  # plt.subplot(122)
  # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target)
  # plt.colorbar()
  # plt.show()

  # digits = load_digits()
  # X = np.vstack([digits.data[digits.target==i]
  #              for i in range(10)])
  # Y = np.hstack([digits.target[digits.target==i]
  #               for i in range(10)])
  # digits_proj = TSNE(random_state=RS).fit_transform(X)
  # # print("type of Y:{}".format(type(Y) ))
  # print("finishe digits_proj!")
  # # scatter(X, Y)
  # scatter(X, Y)

  # plt.savefig('digits_tsne-generated.png', dpi=120)
  # plt.show()


def main():
  run_iris()


if __name__ == "__main__":
    main()
