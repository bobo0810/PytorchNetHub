#!/usr/bin/python
# -*- coding:utf-8 -*-
# power by Mr.Li
import warnings


class DefaultConfig(object):
    env = 'default'  # visdom 环境
    model = 'ResNet34'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    train_data_root = './data/train/'  # 训练集存放路径
    test_data_root = './data/test'  # 测试集存放路径
    load_model_path = './checkpoints/resnet34_0401_11:01:17.pth'  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 128  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 1000
    lr = 0.001  # initial learning rate   初始化学习率bobo
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0  # 损失函数
#命令行来设置配置参数
def parse(self,kwargs):
    for k,v in kwargs.items():
        if not hasattr(self,k):
            warnings.warn("no this pot")
        setattr(self,k,v)

        print("user config:")
        for k,v in  self.__class__.__dict__.items():
            if not k.startswitch('__'):
                print(k,getattr(self,k))

DefaultConfig.parse=parse
#它有自己 和parse的参数
opt=DefaultConfig()
