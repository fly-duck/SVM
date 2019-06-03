#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn import svm
from scipy import stats
from sklearn.metrics import accuracy_score
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys 

def extend(a, b, r=0.01):

    return a * (1 + r) - b * r, -a * r + b * (1 + r)

# main 相当于c语言中的main 就是一个entry point  其实把这段main去掉也没有也没问题
if __name__ == "__main__":
    np.random.seed(0)
    N = 200
    x = np.empty((4*N, 2))
    print(x.shape)  # (800,2)的空矩阵，目的是为了预分配内存,可能会比np.zeros()快一点
    means = [(-1, 1), (1, 1), (1, -1), (-1, -1)]
    sigmas = [np.eye(2), 2*np.eye(2), np.diag((1,2)), np.array(((3, 2), (2, 3)))]
    for i in range(4):
        #多元正态分布
        mn = stats.multivariate_normal(means[i], sigmas[i]*0.1)
        #就是把不同的均值，方差矩阵填充到这个X矩阵中，这个操作x[i*N:(i+1)*N, :]称为切片操作 如果i=0时，就是mn.rvs(N)赋值给x的0到200行
        #切片 https://blog.csdn.net/qq_18433441/article/details/55805619
        x[i*N:(i+1)*N, :] = mn.rvs(N)
        # if(i==0):
            # plt.plot(mn.rvs(N))
        # plt.show()
        #每一个mn.rvs(N)都是（200，2）
        # print(mn.rvs(N).shape)

    #reshape就是改变np数组的形状，看reshape文件 这里就是把行向量转为列向量
    a = np.array((0,1,2,3)).reshape((-1, 1))
    print(a.shape)
    #tile就跟堆积木一样 flatten就是把这个数组变成一维的 看reshape文件 
    y = np.tile(a, N).flatten()

    print(y.shape)
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    #rbf核函数也就是径向基核函数 gamma是核系数，ovr是one vs rest 详情见上面链接
    clf = svm.SVC(C=1, kernel='rbf', gamma=1, decision_function_shape='ovr')
    # clf = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr')
    #拟合模型
    clf.fit(x, y)
    #用这个模型去预测数据的类别 y_hat就是预测的类别
    y_hat = clf.predict(x)
    #比较预测值和真实值，得到正确的概率
    acc = accuracy_score(y, y_hat)
    #设置np的输出格式
    #https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.set_printoptions.html
    # Whether or not suppress printing of small floating point values using scientific notation (default False).
    np.set_printoptions(suppress=True)
    #这个不用多说了，就是定义输出格式，round就是圆整，
    print('预测正确的样本个数：%d，正确率：%.2f%%' % (round(acc*4*N), 100*acc))
    # decision_function
    # print(clf.decision_function(x))
    # print(y_hat)


    #得到每列最大最小值 看reshape
    x1_min, x2_min = np.min(x, axis=0)
    x1_max, x2_max = np.max(x, axis=0)
    # 适当地扩大x1的范围
    x1_min, x1_max = extend(x1_min, x1_max)

    # （1）选出两个维度，x1,x2；选择出其最大最小值（适当的扩大样本已有的数据的范围，使用了extend（自定义的函数））

    # 　　def extend(a, b):
    # 　　　　return 1.05*a-0.05*b, 1.05*b-0.05*a

    # 　　x1_min, x1_max = extend(x[:, 0].min(), x[:, 0].max()) # x1的范围
    # 　　x2_min, x2_max = extend(x[:, 1].min(), x[:, 1].max()) # x2的范围

    # 适当地扩大x2的范围
    x2_min, x2_max = extend(x2_min, x2_max)

    #     >>> np.mgrid[-1:1:5j]
    #array([-1. , -0.5,  0. ,  0.5,  1. ])
    #画网格呗，x1就是 x1min 到 x1max 分500份 x2同理
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
    
    #堆积木，用x_test.shape看他们的维数
    x_test = np.stack((x1.flat, x2.flat), axis=1)
    print(x_test.shape)
    # sys.exit()
    y_test = clf.predict(x_test)
    #改数据形状
    y_test = y_test.reshape(x1.shape)
    #定义两种绘图颜色 其中浅色是区域图，深色是散点图 '#FF8080' 这种16进制的数字就是对应一种颜色 跟'r'一个意思
    cm_light = mpl.colors.ListedColormap(['#FF8080', '#80FF80', '#8080FF', '#F0F080'])
    cm_dark = mpl.colors.ListedColormap(['r', 'g', 'b', 'y'])
    #这里主要是为了中文字体不乱码
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    #定义出一个图片呗
    plt.figure(facecolor='w')
    #画网格图 其实就是个区域图
    plt.pcolormesh(x1, x2, y_test, cmap=cm_light)
    #画分界线
    plt.contour(x1, x2, y_test, levels=(0,1,2), colors='k', linestyles='--')
    #画散点图
    plt.scatter(x[:, 0], x[:, 1], s=20, c=y, cmap=cm_dark, edgecolors='k', alpha=0.7)
    #设置x y 标签
    plt.xlabel('$X_1$', fontsize=11)
    plt.ylabel('$X_2$', fontsize=11)
    #设置坐标轴的范围，跟matlab差不多
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    #定义图片的布局
    plt.grid(b=True)
    plt.tight_layout(pad=2.5)
    #定义图片的title
    plt.title('SVM多分类方法：One/One or One/Other', fontsize=14)
    plt.show()
