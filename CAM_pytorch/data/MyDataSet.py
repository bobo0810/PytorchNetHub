#!/usr/bin/python
# -*- coding:utf-8 -*-
# power by Mr.Li
import os
from torch.utils import data
from torchvision import transforms as T
import cv2
import random
from utils.config import opt
class MyDataSet(data.Dataset):
    '''
    主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
    '''
    def __init__(self, root, transforms=None, train=True, test=False):
        self.test = test  #状态
        self.train = train
        self.root = root  #数据集路径

        # 读取文件夹下所有图像
        if root!='':
            pos_root=os.path.join(root, 'pos')
            neg_root = os.path.join(root, 'neg')

            pos_imgs = [os.path.join(pos_root, img) for img in os.listdir(pos_root)]
            neg_imgs = [os.path.join(neg_root, img) for img in os.listdir(neg_root)]

            imgs = pos_imgs + neg_imgs
            # 打乱数据集
            random.shuffle(imgs)
        else:
            print('数据集为空？？？')
            imgs = []

        imgs_num = len (imgs)
        # 划分数据集
        if train:
            self.imgs = imgs[:int(0.8 * imgs_num)]
        else:
            self.imgs = imgs[int(0.8 * imgs_num):]



        # 对图像进行转化(若未指定转化，则执行默认操作)
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            if self.test or not train:  # 测试集和验证集
                self.transforms = T.Compose([
                    T.ToTensor(),
                    normalize
                ])
            else:  # 训练集
                self.transforms = T.Compose([
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        '''
        一次返回一张图片的数据
        '''
        # 图片的完整路径
        img_path = self.imgs[index]
        # 读取图像
        img = cv2.imread(img_path)
        img = self.BGR2RGB(img)  # 因为pytorch自身提供的预训练好的模型期望的输入是RGB
        img = cv2.resize(img, (64, 128))
        # 对图片进行转化
        img = self.transforms(img)
        # 标签真值
        if 'neg' in img_path:
            label=0  # 没有人
        else:
            label=1   # 有人

        return img,label

    def __len__(self):
        return len(self.imgs)

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_test_img(self):
        # 读取图像
        img_origin = cv2.imread(opt.test_img)
        img = self.BGR2RGB(img_origin)  # 因为pytorch自身提供的预训练好的模型期望的输入是RGB
        img = cv2.resize(img, (64, 128))
        # 对图片进行转化
        img = self.transforms(img)
        return img_origin,img

