"""

1、Python math 模块提供了许多对浮点数的数学运算函数。
exp()是不能直接访问的，需要导入 math 模块，通过静态对象调用该方法。exp() 方法返回x的指数,e^x。

2、

"""

import torch
import torch.nn.functional as F
from math import exp
import numpy as np

"""

结构相似指标可以衡量图片的失真程度，也可以衡量两张图片的相似程度。与MSE和PSNR衡量绝对误差不同，SSIM是感知模型，即更符合人眼的直观感受。
SSIM 主要考量图片的三个关键特征：亮度（Luminance）, 对比度（Contrast）, 结构 (Structure)
亮度以平均灰度衡量，通过平均所有像素的值得到。
对比度通过灰度标准差来衡量。
结构对比可以用相关性系数衡量。

对于图像质量评估，应该局部应用SSIM指数而不是全局应用。
首先，图像统计特征通常是高度空间非平稳的。
其次，图像失真可能取决于或不取决于局部图像统计，也可能是空间变量。
第三，在典型的观测距离下，由于HVS的凹点特征，人眼一次只能看到图像中的具有高分辨率的局部区域。
最后，局部质量测量可以提供一个空间变化的图像质量图，它提供了更多关于图像质量退化的信息，在某些应用中可能有用。

所以，与其在全局范围内应用上述度量值(即一次在图像上的所有区域)，不如在局部范围内应用这些度量值(即在图像的小部分中，然后取整体的平均值)。
所以，SSIM的实践常用方法平均结构相似度指数MSSIM (Mean SSIM)
实际上，当需要衡量一整张图片的质量，经常使用的是以一个一个窗口计算SSIM然后求平均。
当我们用一个一个block去计算平均值，标准差，协方差时，这种方法容易造成 blocking artifacts, 
所以在计算MSSIM时，会使用到 circular-symmetric Gaussian weighting function圆对称的高斯加权公式 
标准差为1.5，和为1，来估计局部平均值，标准差，协方差。

作者使用一个11x11圆对称高斯加权函数(基本上就是一个11x11矩阵，其值来自高斯分布)在整个图像上逐像素移动。
在每一步中，在局部窗口内计算局部统计信息和SSIM索引。
一旦对整个图像进行了计算，我们只需取所有局部SSIM值的平均值，就得到了全局的 SSIM值。

"""

# 这个函数本质上是从一个高斯分布中采样的一个数字列表(长度等于window_size)。所有元素的和等于1，值被归一化。Sigma是高斯分布的标准差。
# 它用于生成上面提到的11x11高斯窗口。
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

# 当我们生成一维高斯张量时，一维张量本身对我们没有用处。因此，我们必须将它转换为一个2D张量(我们之前谈到的11x11张量)。
# 该函数的步骤如下：
# 1、使用高斯函数生成一维张量
# 2、该一维张量与其转置交叉相乘得到二维张量(这保持了高斯特性)
# 3、增加两个额外的维度，将其转换为四维。(仅当SSIM在计算机视觉中用作损失函数时)
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    # x.mm的作用是将参数和变量相乘
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # expand函数的功能就是 用来扩展张量中某维数据的尺寸，它返回输入张量在某维扩展为更大尺寸后的张量。
    # x.contiguous()——把tensor变成在内存中连续分布的形式
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

# 定义结构相似度的计算
# size_average = True，返回 loss.mean()；如果 size_average = False，返回 loss.sum()。
# full：这是布尔类型的可选参数
# 它充当开关并有助于确定返回值的性质。当该值设置为false（默认值）时，仅返回系数；否则，返回0。
# 当该值设置为true时，还将返回来自奇异值分解的诊断信息。
def SSIM(img1, img2, window_size=11, window=None, size_average=True, full=False):
    # 标准化像素的最大值到0-255范围
    img1 = (img1 * 0.5 + 0.5) * 255
    img2 = (img2 * 0.5 + 0.5) * 255
    min_val = 0
    max_val = 255
    L = max_val - min_val
    # torch.clamp(input, min, max, out=None) → Tensor
    # 将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。
    img2 = torch.clamp(img2, 0.0, 255.0)
 
    padd = 0
    # 返回图片的通道数，高度，宽度
    (_, channel, height, width) = img1.size()
    # 如果在函数调用期间没有提供窗口，我们通过 * create_window() * 函数初始化高斯窗口
    if window is None:
        real_size = min(window_size, height, width)
        # 初始化高斯窗口
        window = create_window(real_size, channel=channel).to(img1.device)

    # 计算μ(x)，μ(y)，它们的平方，以及μ(xy)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 计算σ(x), σ(y)和σ(xy)的平方
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
 
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    # 根据提到的公式计算对比度量
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    # 计算SSIM分数并根据公式返回平均值
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
 
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
 
    if full:
        return ret, cs
    return ret

# 为计算PSNR做准备
def tf_log10(x):
    # torch.log是以自然数e为底的对数函数
    numerator = torch.log(x)
    denominator = torch.log(torch.tensor(10.0))
    return numerator / denominator

# 定义峰值信噪比的计算
def PSNR(img1, img2):
    img1 = (img1 * 0.5 + 0.5) * 255
    img2 = (img2 * 0.5 + 0.5) * 255
    max_pixel = 255.0
    img2 = torch.clamp(img2, 0.0, 255.0)
    return 10.0 * tf_log10((max_pixel ** 2) / (torch.mean(torch.pow(img2 - img1, 2))))
