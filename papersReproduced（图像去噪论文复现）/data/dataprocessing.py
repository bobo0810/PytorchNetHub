# -*- coding:utf-8 -*-
# power by Mr.Li
import os
from PIL import  Image
from torch.utils import data
from torchvision import  transforms as T
class DataProcessing(data.Dataset):
    '''
    主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
    '''
    def __init__(self, root, transforms=None, train=True, test=False):
        self.test = test
        self.root=root
        imgs_origin = [os.path.join(root+'JPEGImages/', img) for img in os.listdir(root+'JPEGImages/')]
        imgs_grayscale = [os.path.join(root + 'JPEGImages_bo/', img) for img in os.listdir(root + 'JPEGImages_bo/')]

        if self.test:
            #=====================================
            print()
        else:
            imgs_origin= sorted(imgs_origin, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
            imgs_grayscale = sorted(imgs_grayscale, key=lambda x: int(x.split('.')[-2].split('/')[-1]))

        imgs_num = len(imgs_origin)
        # 2划分数据
        # 如果是测试集就直接用
        if self.test:
            self.imgs_origin = imgs_origin
            self.imgs_grayscale=imgs_grayscale

        elif train:
            self.imgs_origin=imgs_origin[:int(0.7*imgs_num)]
            self.imgs_grayscale = imgs_grayscale[:int(0.7 * imgs_num)]

        else:
            self.imgs_origin=imgs_origin[int(0.7 * imgs_num):]
            self.imgs_grayscale = imgs_grayscale[int(0.7 * imgs_num):]


        # 3 对图像进行转化(若未指定转化，则执行默认操作)===========
        if transforms is None:
            normalize = T.Normalize(mean=[0, 0, 0],
                                    std=[1, 1, 1])

            if self.test or not train:
                self.transforms = T.Compose([
                    T.Scale(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    # T.Scale(256),
                    # T.RandomSizedCrop(224),
                    # T.RandomHorizontalFlip(),
                    # T.ToTensor(),
                    # normalize
                    T.Scale(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
            '''
            一次返回一张图片的数据
            '''
            img_path_origin=self.imgs_origin[index]
            imgs_path_grayscale=self.imgs_grayscale[index]

            if self.test:
                #=================
                print()
            else:
                print()
            #加载一张图片
            data_origin=Image.open(img_path_origin).convert('L')
            data_grayscale=Image.open(imgs_path_grayscale)
            #对图片进行转化
            data_origin=self.transforms(data_origin)
            data_grayscale=self.transforms(data_grayscale)
            return data_origin,data_grayscale
    def __len__(self):
        return len(self.imgs_origin)
