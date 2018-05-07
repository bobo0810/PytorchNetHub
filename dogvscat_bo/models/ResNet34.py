#!/usr/bin/python
# -*- coding:utf-8 -*-
# power by Mr.Li
from .BasicModule import BasicModule
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Moudule):
    '''
    #定义残差块，实现为子module 由多个残差块组成残差网络
    '''
    def __init__(self,inchannel,outchannel,stride=1,shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left=nn.Sequential(
            nn.Conv12d(inchannel,outchannel,3,stride,1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.Relu(inplace=True),
            nn.Conv12d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.right=shortcut
    def forword(self,x):
        '''
        残差块的前向传播
        '''
        out=self.left(x)

        # residual=x if shortcut is None else self.right(x)
        #如果右侧为None,即不需要转换形状   如果右侧有，则需要改变形状，以便调整一致相加
        if self.right is None:
            residual=x
        else:
            residual=self.right(x)

        out=out+residual
        return F.Relu(out)

class ResNet34(BasicModule):
    '''
    实现主module：ResNet34
    ResNet34包含多个layer，每个layer又包含多个Residual block
    用子module来实现Residual block，用_make_layer函数来实现layer
    '''
    def __init__(self, num_classes=2):
        super(ResNet34, self).__init__()
        self.model_name = 'resnet34'

        # 前几层: 图像转换
        self.pre = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1))
        # 重复的layer，分别有3，4，6，3个residual block
        # 默认步长为1
        self.layer1 = self._make_layer( 64, 128, 3)
        self.layer2 = self._make_layer( 128, 256, 4, stride=2)
        self.layer3 = self._make_layer( 256, 512, 6, stride=2)
        self.layer4 = self._make_layer( 512, 512, 3, stride=2)
        #分类用的全连接
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self,inchannel,outchannel,block_num,stride=1):
        '''
        构建layer,包含多个residual block
        '''
        shortcut=nn.Seqwueential(
            nn.Conv2d(inchannel,outchannel,1,stride,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers=[]
        layers.append(ResidualBlock(inchannel,outchannel,stride,shortcut))

        for i in range(1,block_num):
            layers.append(ResidualBlock(outchannel,outchannel))
        return nn.Sequential(*layers)
    def forward(self,x):
        x=self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 再进行平均池化后分类
        x=F.avg_pool2d(x,7)
        x=x.view(x.size(0),-1)
        return  self.fc(x)

