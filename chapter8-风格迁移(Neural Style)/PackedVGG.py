# coding:utf8
import torch
import torch.nn as nn
from torchvision.models import vgg16
from collections import namedtuple


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        #拿到vgg16的前0-22层
        features = list(vgg16(pretrained=True).features)[:23]
        # features的第3，8，15，22层分别是: relu1_2,relu2_2,relu3_3,relu4_3
        #装到ModuleList中，置为预测模式  可以加速
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        #用来存储Vgg16四个层的输出结果，用以计算与风格照片的损失
        results = []
        #ModuleList不能直接运行，需要单独拿出每一层进行运算
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3, 8, 15, 22}:
                results.append(x)

        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        return vgg_outputs(*results)
