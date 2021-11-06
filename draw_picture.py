import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from test_process import *
from attack import *
from load_data import *
import math
import codecs
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def draw(total_epoch, acc_list, mal1_acc_list, mal2_acc_list, MAPE_list, cross_entropy_list, name):
    mnist_test_acc_baseline = np.load('cifar10_test_acc_baseline.npy')
    # 展示测试集、扩充数据集1、扩充数据集2的准确率
    title1 = "the accuracy curve of " + name + " dataset"
    title2 = "the MAPE and cross entropy curve of " + name + " dataset"
    plt.figure()
    X = np.arange(0, total_epoch)
    plt.plot(X, acc_list, label="malicious test accuracy", linestyle=":", linewidth=2)
    plt.plot(X, mal1_acc_list, label="mal1 accuracy", linestyle="--", linewidth=2)
    plt.plot(X, mal2_acc_list, label="mal2 accuracy", linestyle="-.", linewidth=2)
    plt.plot(X, mnist_test_acc_baseline, label="benign test accuracy", linestyle="-", linewidth=2)
    plt.legend()
    plt.title(title1)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()
    plt.close()

    # 展示MAPE 平均交叉熵变化
    plt.figure()
    X = np.arange(0, total_epoch)
    plt.plot(X, MAPE_list, label="MAPE", linestyle=":")
    plt.plot(X, cross_entropy_list, label="cross entropy", linestyle="--")
    plt.legend()
    plt.title(title2)
    plt.xlabel("epoch")
    plt.ylabel("VALUE")
    plt.show()
    plt.close()

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # 原始数据
    x_ = np.arange(10)
    y_ = np.arange(10)
    dz = np.load('result.npy')

    # 柱的位置x,y,z和柱的形状dx,dy,dz
    xx, yy = np.meshgrid(x_ - 0.2, y_ - 0.2)
    x, y, z = xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())
    dx, dy, dz = 0.4 * np.ones_like(z), 0.4 * np.ones_like(z), dz.ravel()
    d = ['b', 'r', 'y', 'darkorange', 'green', 'lightseagreen', 'blue', 'indigo', 'fuchsia', 'slategray']
    c = []
    for i in range(10):
        for j in range(10):
            c.append(d[i])
    # 按原始数据给x,y轴标刻度，并修改刻度的名称
    ax.bar3d(x, y, z, dx, dy, dz, shade=True, color=c)
    plt.xticks(x_)
    plt.yticks(y_)
    ax.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
    ax.set_yticklabels(list('0123456789'))
    ax.set_xlabel('predicted class')
    ax.set_ylabel('label encodings')
    ax.set_zlabel('number')
    plt.title("the output of the second malicious dataset")
    plt.show()


if __name__ == '__main__':
    acc_list = np.load('cifar10_acc.npy')
    mal1_acc_list = np.load('cifar10_mal1_acc.npy')
    mal2_acc_list = np.load('cifar10_mal2_acc.npy')
    MAPE_list = np.load('cifar10_mape.npy')
    cross_entropy_list = np.load('cifar10_cross_entropy.npy')
    mnist_test_acc_baseline = np.load('cifar10_test_acc_baseline.npy')
    total_epoch = len(acc_list)
    draw(total_epoch, acc_list, mal1_acc_list, mal2_acc_list, MAPE_list, cross_entropy_list, 'CIFAR10')