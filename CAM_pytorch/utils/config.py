# -*- coding:utf-8 -*-
# power by Mr.Li
# 设置默认参数
import datetime
import os
class DefaultConfig():
    # 使用的模型，名字必须与models/__init__.py中的名字一致
    # 目前支持的网络
    model = 'VGG16_CAM'

    # 数据集地址
    dataset_root = '/home/bobo/data/cam_dataset/INRIAPerson/Train'

    # 保存模型
    root = os.path.abspath(os.path.join(os.path.dirname(__file__))) + '/'
    checkpoint_root = root + '../checkpoint/'  # 存储模型的路径
    # load_model_path = None  # 加载预训练的模型的路径，为None代表不加载（用于训练）
    load_model_path = checkpoint_root+'VGG16_CAM_39_99.455.pth'

    use_gpu = True  # user GPU or not
    batch_size = 32
    num_workers = 4  #  加载数据时的线程数

    max_epoch = 40


    lr = 0.01
    lr_decay = 0.5

    test_img='/home/bobo/windowsPycharmProject/cam_pytorch/person_and_bike_191.png'  #一张测试图片地址



#初始化该类的一个对象
opt=DefaultConfig()