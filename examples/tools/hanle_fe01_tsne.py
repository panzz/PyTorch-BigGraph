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
from sklearn.decomposition import PCA
import numpy
from numpy import *
import numpy as np
from random import choice

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

def main1():
  filename = '/Users/panzz/Vobs/_dev/ai/_torch/PyTorch-BigGraph/examples/data/fe01/fe01_data.txt'
  outfile = '/Users/panzz/Vobs/_dev/ai/_torch/PyTorch-BigGraph/examples/data/fe01/fe01_output.txt'
  labelfile = '/Users/panzz/Vobs/_dev/ai/_torch/PyTorch-BigGraph/examples/data/fe01/fe01_label.txt'
  nheader = '/m/'
  theader = '/frontend/'
  try:
    # with open(outfile, "w") as f2:
    #     f2.write('')
    with open(outfile, 'r+', encoding='utf-8') as f:
        res = f.readlines()
        print(res)
        f.seek(0)
        f.truncate()
    with open(labelfile, 'r+', encoding='utf-8') as f:
        res = f.readlines()
        print(res)
        f.seek(0)
        f.truncate()

    with open(filename, "r") as f:
        lines = f.readlines()
        print(len(lines), type(lines))
        for l in lines:
          nll = l.split('\t')
          print('nll[{}]:{}'.format(type(nll), len(nll)))
          if nll[0].find(nheader) >= 0 or nll[0].find(theader) >= 0:
            with open(labelfile, 'a') as f2:
              f2.write(nll[0]+'\n')
            del nll[0]
          else:
            print('lines[{}]:{}'.format(lines.index(l), l))
          nllstr = '\t'.join(nll)
          
          with open(outfile, 'a') as f3:
            f3.write(nllstr)

          # with open(outfile, "a") as f1:
          #   f1.write(nllstr)
  except:
      print("打开文件异常")

fdata_test= '/Users/panzz/Vobs/_dev/ai/_torch/PyTorch-BigGraph/examples/data/fe01/3.txt'
ftarget_test= '/Users/panzz/Vobs/_dev/ai/_torch/PyTorch-BigGraph/examples/data/fe01/4.txt'

fdata= '/Users/panzz/Vobs/_dev/ai/_torch/PyTorch-BigGraph/examples/data/fe01/fe01_output_test.tsv' # "3.txt"
ftarget= '/Users/panzz/Vobs/_dev/ai/_torch/PyTorch-BigGraph/examples/data/fe01/fe01_label_test.tsv' # "4.txt"   

class chj_data_test(object):
    def __init__(self,data,target):
        self.data=data
        self.target=target

def chj_load_file_test(fdata,ftarget):
    data=numpy.loadtxt(fdata, dtype=float32)
    target=numpy.loadtxt(ftarget, int32)

    print(data.shape)
    print(target.shape)
    # pexit()

    res=chj_data_test(data,target)
    return res

class chj_data(object):
    def __init__(self,data,target):
        self.data=data
        self.target=target

def chj_load_file(fdata,ftarget):
    data=numpy.loadtxt(fdata, delimiter='\t', dtype=float32)
    target=numpy.loadtxt(ftarget, dtype='str')#int32)

    print(data.shape)
    print(target.shape)
    # pexit()

    res=chj_data(data,target)
    return res

def run_iris():
  iris = load_iris() # 使用sklearn自带的测试文件
  # iris = chj_load_file_test(fdata_test, ftarget_test)
  # iris = chj_load_file(fdata, ftarget)

  X_tsne = TSNE(n_components=2,learning_rate=100).fit_transform(iris.data)
  # X_pca = PCA().fit_transform(iris.data)
  print("finishe!")
  plt.figure(figsize=(12, 6))
  # plt.subplot(121)
  plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=iris.target)
  # plt.subplot(122)
  # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target)
  plt.colorbar()
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


class RandomWalk(object):
    """一个生成随机漫步数据的类"""

    def __init__(self, num_points=5000):
        """初始化随机漫步的属性"""
        #存储随机漫步次数的变量
        self.num_points = num_points
        #所有随机漫步都始于(0,0)
        #分别存储随机漫步经过的每个点的x和y坐标
        self.x_values = [0]
        self.y_values = [0]

    def fill_walk(self):
        """计算随机漫步包含的所有点"""

        #不断漫步，直到列表达到指定的长度
        while len(self.x_values) < self.num_points:
            #决定前进方向以及沿这个方向前进的距离
            x_direction = choice([1, -1])
            x_distance = choice([0, 1, 2, 3, 4])
            x_step = x_direction * x_distance

            y_direction = choice([1, -1])
            y_distance = choice([0, 1, 2, 3, 4])
            y_step = y_direction * y_distance

            #拒绝原地踏步
            if x_step == 0 and y_step == 0:
                continue

            #计算下一个点的x值和y值
            next_x = self.x_values[-1] + x_step
            next_y = self.y_values[-1] + y_step

            self.x_values.append(next_x)
            self.y_values.append(next_y)
        pass

def rand_walk():
  rw = RandomWalk()
  rw.fill_walk()
  point_numbers = list(range(rw.num_points))

  plt.scatter(rw.x_values, rw.y_values, c=point_numbers, cmap=plt.cm.Blues, edgecolor='none', s=15)
  plt.show()


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
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

def _joint_probabilities_constant_sigma(D, sigma):
    P = np.exp(-D**2/2 * sigma**2)
    P /= np.sum(P, axis=1)
    return P


def _gradient_descent(objective, p0, it, n_iter, n_iter_without_progress=30,
                      momentum=0.5, learning_rate=1000.0, min_gain=0.01,
                      min_grad_norm=1e-7, min_error_diff=1e-7, verbose=0,
                      args=[]):
    # The documentation of this function can be found in scikit-learn's code.
    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(np.float).max
    best_error = np.finfo(np.float).max
    best_iter = 0

    for i in range(it, n_iter):
        # We save the current position.
        positions.append(p.copy())

        new_error, grad = objective(p, *args)
        error_diff = np.abs(new_error - error)
        error = new_error
        grad_norm = linalg.norm(grad)

        if error < best_error:
            best_error = error
            best_iter = i
        elif i - best_iter > n_iter_without_progress:
            break
        if min_grad_norm >= grad_norm:
            break
        if min_error_diff >= error_diff:
            break

        inc = update * grad >= 0.0
        dec = np.invert(inc)
        gains[inc] += 0.05
        gains[dec] *= 0.95
        np.clip(gains, min_gain, np.inf)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

    return p, error, i


RS = 20150101
# This list will contain the positions of the map points at every iteration.
positions = []
def tsne_test():
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})

    digits = load_digits()
    digits.data.shape
    # print(digits['DESCR'])

    nrows, ncols = 2, 5
    plt.figure(figsize=(6,3))
    plt.gray()
    for i in range(ncols * nrows):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.matshow(digits.images[i,...])
        plt.xticks([]); plt.yticks([])
        plt.title(digits.target[i])
    # plt.savefig('digits-generated.png', dpi=150)
    X = np.vstack([digits.data[digits.target==i]
               for i in range(10)])
    y = np.hstack([digits.target[digits.target==i]
               for i in range(10)])
    digits_proj = TSNE(random_state=RS).fit_transform(X)   
    scatter(digits_proj, y)
    # plt.savefig('digits_tsne-generated.png', dpi=120)

    # Pairwise distances between all data points.
    D = pairwise_distances(X, squared=True)
    # Similarity with constant sigma.
    P_constant = _joint_probabilities_constant_sigma(D, .002)
    # Similarity with variable sigma.
    P_binary = _joint_probabilities(D, 30., False)
    # The output of this function needs to be reshaped to a square matrix.
    P_binary_s = squareform(P_binary)

    plt.figure(figsize=(12, 4))
    pal = sns.light_palette("blue", as_cmap=True)

    plt.subplot(131)
    plt.imshow(D[::10, ::10], interpolation='none', cmap=pal)
    plt.axis('off')
    plt.title("Distance matrix", fontdict={'fontsize': 16})

    plt.subplot(132)
    plt.imshow(P_constant[::10, ::10], interpolation='none', cmap=pal)
    plt.axis('off')
    plt.title("$p_{j|i}$ (constant $\sigma$)", fontdict={'fontsize': 16})

    plt.subplot(133)
    plt.imshow(P_binary_s[::10, ::10], interpolation='none', cmap=pal)
    plt.axis('off')
    plt.title("$p_{j|i}$ (variable $\sigma$)", fontdict={'fontsize': 16})
    plt.savefig('similarity-generated.png', dpi=120)
    sklearn.manifold.t_sne._gradient_descent = _gradient_descent
    plt.show()


def main():
  # run_iris()
  # rand_walk()
  tsne_test()


if __name__ == "__main__":
    main()
    
