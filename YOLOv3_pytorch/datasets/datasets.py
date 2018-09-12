import glob
import random
import os
import numpy as np

import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.transform import resize

import sys

class ImageFolder(Dataset):
    '''
    仅detect.py用到，用于测试例子
    '''
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = np.array(Image.open(img_path))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        # 返回图片路径、经过处理后的图像tensor
        return img_path, input_img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    '''
    数据集加载器
    '''
    def __init__(self, list_path, img_size=416):
        # 读取  数据集中分配为训练集的txt文本，以list形式保存
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        # 读取  数据集中分配为训练集的txt文本（即标签，coco数据集以txt保存 框真值），以list形式保存
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        # 输入训练图像大小
        self.img_shape = (img_size, img_size)
        self.max_objects = 50  # 设定一张图像最多真实存在50个物体（封装 图像真值框时使用到）

    def __getitem__(self, index):

        '''
        训练时获取单张图像及真值
        '''

        # 读取图像为tensor
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = np.array(Image.open(img_path))

        # Handles images with less than three channels
        # 处理 图像的通道数不为3 时(即该图像损坏)，则 读取下一张图片
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        # 对图像tensor进行处理（数据增强、规范化）

        # w,h按照较大值填充成正方形
        h, w, _ = img.shape
        # np.abs 绝对值
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        # 上（左）和下（右）填充
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding 确定填充
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding 添加填充
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.

        # 填充成正方形后 resize到 指定形状（一般为416x416）
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize  resize并规范化
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')

        # Channels-first  转换通道
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor  转为pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        #---------
        #  真值标签处理
        #---------
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        labels = None
        if os.path.exists(label_path):
            # eg：[8,5]   8：该图像有8个bbox   5: 0代表类别对应序号 1~4代表坐标（值在0~1之间）
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image
            # 提取未填充+未缩放图像的坐标
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)
            # Adjust for added padding
            # 添加填充，以便于 图像调整一致
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            # Calculate ratios from coordinates
            # 从坐标计算比率
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
        # Fill matrix
        # 填充矩阵（将 txt里的内容，即每张图像的所有物体填入，最多添加50个物体）
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)
        # 返回 图像路径、处理后的图像tensor、坐标被归一化后的真值框filled_labels[50,5] 值在0-1之间
        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)
