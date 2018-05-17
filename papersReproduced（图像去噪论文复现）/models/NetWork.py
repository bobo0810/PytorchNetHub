# -*- coding:utf-8 -*-
# power by Mr.Li

from torch import nn
import  time
import torch as t
class NetWork(nn.Module):
    '''
    定义网络
    '''
    def __init__(self):
        super(NetWork,self).__init__()
        self.main=nn.Sequential(

            #问题：padding 考虑   无pool？  LeakyReLU更优？ ？？？？？？？？？？？？？？？
            # 1代表输入时的图像层数,32代表输出的层数,5代表卷积核
            # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            #卷积
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5,stride=1,padding=2,bias=True),
            #通道数BN层的参数是通道数out_channels=32
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            # 通道数BN层的参数是通道数out_channels=64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
            # 通道数BN层的参数是通道数out_channels=64
            nn.BatchNorm2d(64),
            nn.ReLU(True),


            #反卷积
            # class torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True)
            nn.ConvTranspose2d(in_channels=64,out_channels=32 ,kernel_size=3, stride=1,padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=2, bias=True),
            # nn.Tanh()  # 输出范围 -1~1 故而采用Tanh # 输出形状：1x 长 x 宽
            nn.Sigmoid()  # 输出范围 0~1
        )
    def forward(self,x):
        x=self.main(x)
        return x

    # def save(self, name=None):
    #     if name is None:
    #         prefix = '/home/bobo/PycharmProjects/torchProjectss/papersReproduced/checkpoints/' + self.moduel_name + "_"
    #         name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
    #     t.save(self.stat_dict(), name)
    #     return name