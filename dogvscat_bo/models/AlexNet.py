#!/usr/bin/python
# -*- coding:utf-8 -*-
# power by Mr.Li

from torch import nn
from .BasicMoudle improt BasicModule

class AlexNet(BasicModule):
    def __init__(self,num_claasses=2):
        super(AlexNet, self).__init__()
        self.model_name = 'alexnet'
        self.feature=nn.Sequential(
            nn.Conv2d(3,64,kernel_Size=11,stride=4,padding=2),
            nn.RELU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(64, 192, kernel_Size=5,  padding=2),
            nn.RELU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d( 384,256,kernel_size=3,stride=1),
            nn.RELU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.RELU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )
        self.classfier=nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6,4096),
            nn.RELU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.RELU(inplace=True),
            #nn.Dropout(),
            nn.Linear(4096,num_claasses),
            #nn.RELU()
        )
        def forward(self,x):
            x=self.feature(x),
            #特征层到分类层需要调整形状   由多通道展开为单通道
            x=x.view(x.size(0),256*6*6),
            x=self.classfier(x)
            return x

