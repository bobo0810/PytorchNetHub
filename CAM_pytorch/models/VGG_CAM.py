# -*- coding:utf-8 -*-
# power by Mr.Li
from torch import nn
import torch as t
from torchvision.models import vgg16
from utils.config import opt
class VGG16_CAM(nn.Module):
    '''
    定义网络
    '''
    def __init__(self):
        super(VGG16_CAM, self).__init__()
        # 设置网络名称
        self.moduel_name = str("VGG16_CAM")
        # 去掉 VGG16 feature层的maxpool层
        self.feature_layer = nn.Sequential(*list(vgg16(pretrained=True).features.children())[0:-1])
        # 全局平均池化层 GAP
        self.fc_layer = nn.Linear(512,2)

    def forward(self, x):
        x = self.feature_layer(x)
        # GAP 全局平均池化
        x = t.mean(x,dim=3)
        x = t.mean(x,dim=2)

        # 全连接层+softmax层
        x = self.fc_layer(x)
        # x = F.softmax(x)   #交叉熵自带softmax
        return x


# def test():
#     from torch.autograd import Variable
#     model=VGG16_CAM()
#     print(model)
#     img=t.rand(2,3,224,224)
#     img=Variable(img)
#     output=model(img)
#     print(output.size())
#
# if __name__ == '__main__':
#     test()