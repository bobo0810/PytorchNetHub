#!/usr/bin/python
# -*- coding:utf-8 -*-
# power by Mr.Li
import os
from PIL import  Image
from torch.utils import data
import numpy as np
from torchvision import  transforms as T

class DogCat(data.Dataset):
    '''
    主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
    '''
    def __init__(self, root, transforms=None, train=True, test=False):
        self.test=test
        # os.path.join(root, img)合并两个字符串，拼接为一个字符串
        imgs=[os.path.join(root,img) for img in os.listdir(root)]
        #1加载图片名称
        # test1: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg
        # 测试集图片名称只有数字
        if self.test:
            # data/test1/8973.jpg   x.spli('.')[-2]后为data/test1/8973   split('/')[-1]后为8973
            imgs=sorted(imgs,key=lambda x:int(x.split('.')[-2].split('/')[-1]))
        else:
            # train: data / train / cat.10004.jpg   x.split('.')[-2]后为10004    [-2]为倒数第二个
            imgs=sorted(imgs,key=lambda x:int(x.split('.')[-2]))
        imgs_num = len(imgs)
        #2划分数据
        #如果是测试集就直接用
        if self.test:
            self.imgs = imgs
        #否则是训练集和验证集。需要划分训练集0到0.7和  验证集 为0.到1
        elif train:
            self.imgs = imgs[:int(0.7*imgs_num)]
        else :
            self.imgs = imgs[int(0.7*imgs_num):]

        #3 对图像进行转化(若未指定转化，则执行默认操作)
        #mean 和 std是Image数据集所有图片进行计算的，符合大自然规律
        if transforms is None:
            normalize=T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
            #测试集 和 验证集的转换
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Scale(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                # 训练集进行数据增强
                self.transforms = T.Compose([
                    T.Scale(256),
                    T.RandomSizedCrop(224), #随机裁剪
                    T.RandomHorizontalFlip(),  #随机水平翻转
                    T.ToTensor(),
                    normalize
                    ])
        def __getitem__(self, index):
            '''
            一次返回一张图片的数据
            '''
            img_path=self.imgs[index]
            #读取图片对应的标签
            if self.test: #测试集命名只有数字  data/test1/8973.jpg    label为8973
                label = int(self.imgs[index].split('.')[-2].split('/')[-1])
            else:  #训练集和验证集 dog:1  cat:0
                label = 1 if 'dog' in img_path.split('/')[-1] else 0
            #加载一张图片
            data=Image.open(img_path)
            #对图片进行转化
            data=self.transforms(data)
             return data,label
        def __len__(self):
             return len(self.imgs)
