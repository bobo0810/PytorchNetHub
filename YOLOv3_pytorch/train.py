from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')  # 训练轮数
parser.add_argument('--image_folder', type=str, default='data/samples', help='path to dataset') #数据集地址
parser.add_argument('--batch_size', type=int, default=16, help='size of each image batch')  #batch大小
parser.add_argument('--model_config_path', type=str, default='config/yolov3.cfg', help='path to model config file') # 模型网络结构
parser.add_argument('--data_config_path', type=str, default='config/coco.data', help='path to data config file') # 配置数据集的使用情况
parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')  # 网络权重
parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file') #coco数据集类别标签
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold') # 物体置信度阈值
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression') # iou for nms的阈值
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation') # 批生成期间要使用的cpu线程数
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')   # 输入图像尺寸的大小
parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between saving model weights') # 每隔几个模型保存一次
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='directory where model checkpoints are saved') # 保存生成模型的路径
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available') # 是否使用GPU
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs('output', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

classes = load_classes(opt.class_path) #coco数据集类别标签

# Get data configuration
# 获取dataloader配置
data_config     = parse_data_config(opt.data_config_path)
# 拿到训练集
train_path      = data_config['train']

# Get hyper parameters

#hyperparams 即cfg中的[net]部分，网络训练的超参数
hyperparams     = parse_model_config(opt.model_config_path)[0]   # model_config_path：模型网络结构cf文件
learning_rate   = float(hyperparams['learning_rate'])
momentum        = float(hyperparams['momentum'])
decay           = float(hyperparams['decay'])
burn_in         = int(hyperparams['burn_in'])

# Initiate model
# 初始化模型
model = Darknet(opt.model_config_path) # model_config_path：模型网络结构
#model.load_weights(opt.weights_path)
#初始化Conv、BatchNorm2d权重
model.apply(weights_init_normal)

if cuda:
    model = model.cuda()

# 将模型调整为训练模式
model.train()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path),
    batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)


# 设置好 默认新建的tensor类型
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# 优化器
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=0, weight_decay=decay)

# 开始训练
for epoch in range(opt.epochs):
    # 每轮epoch
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        # imgs :处理后的图像tensor[16,3,416,416]        targets:坐标被归一化后的真值框filled_labels[16,50,5] 值在0-1之间
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)
        # 优化器梯度清零
        optimizer.zero_grad()
        # 得到网络输出值，作为损失 (loss :多尺度预测的总loss之和)
        loss = model(imgs, targets)
        # 反向传播  自动求梯度
        loss.backward()
        # 更新优化器的可学习参数
        optimizer.step()

        print('[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f]' %
                                    (epoch, opt.epochs, batch_i, len(dataloader),
                                    model.losses['x'], model.losses['y'], model.losses['w'],
                                    model.losses['h'], model.losses['conf'], model.losses['cls'],
                                    loss.item(), model.losses['recall']))

        # 统计 训练过程共使用多少张图片，用于 保存权重时写入 头文件中
        model.seen += imgs.size(0)
    # 每隔几个模型保存一次
    if epoch % opt.checkpoint_interval == 0:
        model.save_weights('%s/%d.weights' % (opt.checkpoint_dir, epoch))
