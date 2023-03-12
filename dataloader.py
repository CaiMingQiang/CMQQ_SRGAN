from random import randint

import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

from .utils import cvtColor, preprocess_input

# 训练时，输入的高分辨率图像一般为很大的图片。需要将其随机裁剪为预设的大小。再将裁剪的图像，下采样作为低分辨率图像。

# 重设图片尺寸为正方形
def get_new_img_size(width, height, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_width, resized_height

# 定义数据读取类
class SRGANDataset(Dataset):
    # 类构造函数
    def __init__(self, train_lines, lr_shape, hr_shape):
        # 继承父类
        super(SRGANDataset, self).__init__()
        # 数据集的长度
        self.train_lines    = train_lines
        # 总batch数量
        self.train_batches  = len(train_lines)

        self.lr_shape       = lr_shape
        self.hr_shape       = hr_shape
    # 获取数据集大小
    def __len__(self):
        # 返回数据集大小
        return self.train_batches
    # 根据索引提取数据
    def __getitem__(self, index):
        index = index % self.train_batches
        # Python中split是一个内置函数，用来对字符串进行分割，分割后的字符串以列表形式返回
        # 将train_line按空格分割为line列表， 并根据索引获取原始图像
        image_origin = Image.open(self.train_lines[index].split()[0])
        if self.rand()<.5:
            img_h = self.get_random_data(image_origin, self.hr_shape)
        else:
            img_h = self.random_crop(image_origin, self.hr_shape[1], self.hr_shape[0])
        # 利用双三次插值对HR图像进行下采样，得到LR图像
        img_l = img_h.resize((self.lr_shape[1], self.lr_shape[0]), Image.BICUBIC)
        # 对HR,LR图像标准化，并对通道进行调换
        img_h = np.transpose(preprocess_input(np.array(img_h, dtype=np.float32), [0.5,0.5,0.5], [0.5,0.5,0.5]), [2,0,1])
        img_l = np.transpose(preprocess_input(np.array(img_l, dtype=np.float32), [0.5,0.5,0.5], [0.5,0.5,0.5]), [2,0,1])
        # 返回HR，LR图像
        return np.array(img_l), np.array(img_h)

    # 返回0-1之间随机数
    def rand(self, a=0, b=1):
        # np.random.rand()返回0-1之间的随机数
        return np.random.rand()*(b-a) + a

    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        # 真实输入图像大小
        # 图片的宽和高，iw和ih
        iw, ih  = image.size
        # 网络输入大小
        # 输入尺寸的高和宽，h和w
        h, w    = input_shape
        # 如果是非随机的，将图片等比例转化为网络输入大小
        if not random:
            # 保证长或宽，符合目标图像的尺寸
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            # 采用双三次插值算法缩小图像
            image       = image.resize((nw,nh), Image.BICUBIC)
            # 创建一个新的灰度图像，缩放后的图像可能不能满足网络大小，所以可以给周边补充一些灰度条。
            # 其余用灰色填充，即(128, 128, 128)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            # 将灰度条和裁剪后图合在一起
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            return image_data

        """
        
        计算机视觉方面，计算机视觉的主要问题是没有办法得到充足的数据。
        对大多数机器学习应用，这不是问题，但是对计算机视觉，数据就远远不够。
        所以这就意味着当你训练计算机视觉模型的时候，数据增强会有所帮助，这是可行的。
        
        一般的数据增强方式：
        1、对图像进行缩放并进行长和宽的扭曲
        2、对图像进行翻转
        3、对图像进行色域扭曲
        
        """

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#

        # 调整图片大小
        new_ar = w/h * self.rand(1-jitter,1+jitter)/self.rand(1-jitter,1+jitter)
        scale = self.rand(1, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        # 放置图片

        """
        
        放置图片这一步其实还有一个叫法叫添加灰条，其目的是让所有的输入图片都是同样大小的正方形图片，
        因为上面的图片是做个缩放的，所以大小不能确定，大小不够的就需要添加灰条，
        这里就可以简单理解为放了一张正方形白纸在那里，你可以在纸上随便画你想要的东西，填不满的就保留白纸，
        最终训练的时候也是将这张白纸直接输入，这样就能实现输入图像都是一样大的，但是图内物体的大小却不同。
        
        """

        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        rotate = self.rand()<.5
        if rotate: 
            angle = np.random.randint(-15,15)
            a,b = w/2,h/2
            M = cv2.getRotationMatrix2D((a,b),angle,1)
            image = cv2.warpAffine(np.array(image), M, (w,h), borderValue=[128,128,128]) 

        #------------------------------------------#
        #   色域扭曲
        #------------------------------------------#
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
        return Image.fromarray(np.uint8(image_data))

    def random_crop(self, image, width, height):
        #--------------------------------------------#
        #   如果图像过小无法截取，先对图像进行放大
        #--------------------------------------------#
        if image.size[0] < self.hr_shape[1] or image.size[1] < self.hr_shape[0]:
            resized_width, resized_height = get_new_img_size(width, height, img_min_side=np.max(self.hr_shape))
            image = image.resize((resized_width, resized_height), Image.BICUBIC)

        #--------------------------------------------#
        #   随机截取一部分
        #--------------------------------------------#
        width1  = randint(0, image.size[0] - width)
        height1 = randint(0, image.size[1] - height)

        width2  = width1 + width
        height2 = height1 + height

        image   = image.crop((width1, height1, width2, height2))
        return image
        
def SRGAN_dataset_collate(batch):
    images_l = []
    images_h = []
    for img_l, img_h in batch:
        images_l.append(img_l)
        images_h.append(img_h)
    return np.array(images_l), np.array(images_h)
