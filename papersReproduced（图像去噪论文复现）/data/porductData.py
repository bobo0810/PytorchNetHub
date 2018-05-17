import glob as gb
from PIL import Image

from pylab import *

from numpy import *
import ipdb
import random

img_path = gb.glob("/home/bobo/data/VOCdevkit/Pascal VOC2007/VOCdevkit/VOC2007/JPEGImages/*.jpg")

ii = 0;
for path in img_path:
    # 只拿15000张照片  其实只有5000张照片

    if (ii < 15000):
        im = array(Image.open(path).convert('L'))
        # 设定高斯函数的偏移
        means = 0
        # 设定高斯函数的标准差
        sigma = 25
        im_flatten = im[:, :].flatten()
        # 计算新的像素值
        for i in range(im.shape[0] * im.shape[1]):
            pr = int(im_flatten[i]) + random.gauss(0, sigma)
            if (pr < 0):
                pr = 0
            if (pr > 255):
                pr = 255
            im_flatten[i] = pr
        im[:, :] = im_flatten.reshape([im.shape[0], im.shape[1]])
        # 保存灰度图
        from scipy import misc

        # 拿到照片名字
        imgs_name = path.split('.')[-2].split('/')[-1]

        misc.imsave(
            '/home/bobo/data/VOCdevkit/Pascal VOC2007/VOCdevkit/VOC2007/JPEGImages_Noise_added_grayscale/' + imgs_name + '.jpg',
            im)
img_path2 = gb.glob("/home/bobo/data/VOCdevkit/Pascal VOC2007/VOCdevkit/VOC2007/JPEGImages_Noise_added_grayscale/*.jpg")
print(img_path2.__len__())