"""

itertools,是python的一个内置模块，功能强大，主要用于高效循环创建迭代器。

"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch

# 定义展示结果的函数
def show_result(num_epoch, G_net, imgs_lr, imgs_hr):
    # 当前计算不需要反向传播
    with torch.no_grad():
        # 将lr图片传入生成器
        test_images = G_net(imgs_lr)
        # 一次性在画布上创建1*2的网格
        fig, ax = plt.subplots(1, 2)
        # 创建长度为2的迭代器
        for j in itertools.product(range(2)):
            # 将坐标轴的可见性设置为False，从而隐藏坐标轴
            ax[j].get_xaxis().set_visible(False)
            ax[j].get_yaxis().set_visible(False)
        # cla函数的作用是清除坐标区
        ax[0].cla()
        # 画布中左图展示生成器生成的图片tet_images
        ax[0].imshow(np.transpose(test_images.cpu().numpy()[0] * 0.5 + 0.5, [1,2,0]))

        ax[1].cla()
        # 画布中右图展示原始hr图片
        ax[1].imshow(np.transpose(imgs_hr.cpu().numpy()[0] * 0.5 + 0.5, [1,2,0]))
        # 设置图片标签
        label = 'Epoch {0}'.format(num_epoch)
        # 调整图片标签的位置
        fig.text(0.5, 0.04, label, ha='center')
        # 保存结果到指定路径并命名
        plt.savefig("results/train_out/epoch_" + str(num_epoch) + "_results.png")
        # 关闭当前窗口
        plt.close('all')  # 避免内存泄漏


"""

#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB

"""

# 定义图像类型转换函数
def cvtColor(image):
    # 如果图片是3维数组.同时第3维度表示通道数为3，则返回原有图像
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    # 否则转换为RGB图像
    else:
        image = image.convert('RGB')
        return image 

# 对原始输入进行预处理
def preprocess_input(image, mean, std):
    # 将原始图像标准化到0-1区间内
    image = (image/255 - mean)/std
    return image

# 定义动态修改学习率的函数
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']