import numpy as np
import os
import torch.utils.data as data
import torch
import random
import glob
from tqdm import tqdm
from timm.data.transforms_factory import create_transform as timm_transform
from PIL import Image
import torch
import cv2
import os
import numpy as np
import torchvision
from torchvision.transforms import transforms

def Process(img_path, img_size, use_augment):
    """
    timm默认预处理
    """
    # 读取图像
    assert os.path.exists(img_path), f"{img_path} 图像不存在"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR
    img = Image.fromarray(img)
    if use_augment:
        # 增广：Random(缩放、裁剪、翻转、色彩...)
        img_trans = timm_transform(
            img_size,
            is_training=True,
            re_prob=0.5,
            re_mode="pixel",  # 随机擦除
            auto_augment=None,  # 自动增广  eg：rand-m9-mstd0.5
        )
    else:
        # 不增广：ReSize256 -> CenterCrop224
        img_trans = timm_transform(img_size)
    return img_trans(img)


class BBN_Dataset(data.Dataset):
    """数据加载器"""

    def __init__(self, txt_path, mode, size):
        """

        Args:
            txt_path (str): 数据集路径
            mode (str): 类型  
            size (list): 图像尺寸  eg: [224,224]
        """
        assert mode in ["train", "val", "test"]
        self.use_augment = True if mode == "train" else False  # 训练集开启增广
        self.size = size

        self.dataset = self.load_txt(txt_path)
        self.imgs_list = self.dataset[mode]
        self.all_labels = self.dataset["all_labels"]

        # 训练集开启BBN
        if mode == "train":

            labels_list = [label for _, label in self.imgs_list]  # 所有图片对应的类别列表
            class_index_dict = dict()  # key类别名对应的索引  values该类的所有图片索引
            class_nums_list = [0] * len(self.all_labels)  # 每个类对应的图片数
            for index, label in enumerate(labels_list):
                if not int(label) in class_index_dict:
                    class_index_dict[int(label)] = []
                class_index_dict[int(label)].append(index)

                class_nums_list[int(label)] += 1

            # 构建逆向采样分布
            max_num = max(class_nums_list)  # 类内最大样本数
            class_weight = [max_num / i for i in class_nums_list]  # 概率占比的倒数 列表
            sum_weight = sum(class_weight)  # 逆向的概率占比之和
            self.class_weight, self.sum_weight = class_weight, sum_weight
            self.class_index_dict = class_index_dict

    def __getitem__(self, index):
        img_path, label = self.imgs_list[index]
        # 图像预处理
        img = Process(img_path, self.size, self.use_augment)
        # 训练集 BBN采样
        if self.use_augment:
            sample_class = self.sample_class_index_by_weight()  # 类别索引
            sample_indexes = self.class_index_dict[sample_class]  # 获得该类别的所有图片索引（对应图片顺序）
            sample_index = random.choice(sample_indexes)  # 随机抽取一个样本
            img2_path, label2 = self.imgs_list[sample_index]
            img2 = Process(img2_path, self.size, self.use_augment)

            return img, label, img_path, img2, label2, img2_path
        # 验证集/测试集
        else:
            return img,label,img_path

    def __len__(self):
        return len(self.imgs_list)

    def load_txt(self, txt_path):
        """单标签分类 加载数据集

        Args:
            txt_path (str): 数据集路径

            训练格式形如  类型, 类别名, 图片路径
            train, dog,  img1.jpg
            val,   dog,  img2.jpg
            test,  cat,  img3.jpg

            返回:
            {
                "train": [
                    img_1, 0,
                    img_2, 1,
                    ...
                ],
                "val":  类似,
                "test": 类似,
                "all_labels": ["dog", "cat",...],
            }

        """
        # 读取
        f = open(txt_path)
        txt_list = f.readlines()
        txt_list =[ txt.split(",")  for txt in txt_list]
        f.close()
        
        
        # 获取所有类别
        all_labels = [txt_i[1] for txt_i in txt_list]
        all_labels = list(set(all_labels))
        all_labels.sort()

        # 构建数据集
        dataset = {
            "train": [],
            "val": [],
            "test": [],
            "all_labels": all_labels,
        }
        for mode, label, img_path in txt_list:
            assert mode in ["train", "val", "test"]
            dataset[mode].append([img_path, all_labels.index(label)])
        return dataset

    def sample_class_index_by_weight(self):
        """
        逆向采样
        """
        # rand_number  0~逆向比例之和
        rand_number, now_sum = random.random() * self.sum_weight, 0
        # self.cls_num 类别总数
        # 遍历每个类别   即判断随机数处于哪个类别范围内，即返回该类别索引
        for i in range(len(self.class_weight)):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i  # 采样的类别索引