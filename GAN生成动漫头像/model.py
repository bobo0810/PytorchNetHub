# coding:utf8
from torch import nn


class NetG(nn.Module):
    """
    生成器定义
    """

    def __init__(self, opt):
        super(NetG, self).__init__()
        #即刚开始生成多少层（多少通道）的feature map数
        #最后结果是要生成3层feature map（即RGB三个通道）
        ngf = opt.ngf  # 生成器feature map数

        self.main = nn.Sequential(
            # Hout = (H in -1) * stride - 2 * padding + kernel_size
            # padding为0或1则Hout对Hin做除法，商为步长，余数为卷积核大小
            # 输入是一个nz维度的噪声，我们可以认为它是一个1*1*nz的feature map
            #class torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True)
            # in_channels(int) – 输入信号的通道数
            # out_channels(int) – 卷积产生的通道数
            # kerner_size(int or tuple) - 卷积核的大小
            # stride(int or tuple, optional) - 卷积步长
            # padding(int or tuple, optional) - 输入的每一条边补充0的层数
            # output_padding(int or tuple, optional) - 输出的每一条边补充0的层数
            # dilation(int or tuple, optional) – 卷积核元素之间的间距
            # groups(int, optional) – 从输入通道到输出通道的阻塞连接数
            # bias(bool, optional) - 如果bias = True，添加偏置
            nn.ConvTranspose2d(opt.nz, ngf * 8, 4, 1, 0, bias=False),
            #BatchNorm2d是对一个batch计算均值 方差 来进行规范化
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            #输出的通道数为(ngf*8)   大小为为 通过公式计算
            # 上一步的输出形状：(ngf*8) x 4 x 4

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 上一步的输出形状： (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 上一步的输出形状： (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 上一步的输出形状：(ngf) x 32 x 32

            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            nn.Tanh()  # 输出范围 -1~1 故而采用Tanh
            # 输出形状：3 x 96 x 96
        )

    def forward(self, input):
        return self.main(input)


class NetD(nn.Module):
    """
    判别器定义
    """

    def __init__(self, opt):
        super(NetD, self).__init__()
        ndf = opt.ndf
        self.main = nn.Sequential(

            #nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            #in_channels(int) – 输入信号的通道
            # out_channels(int) – 卷积产生的通道
            # kerner_size(int or tuple) - 卷积核的尺寸
            # stride(int or tuple, optional) - 卷积步长
            # padding(int or tuple, optional) - 输入的每一条边补充0的层数
            # dilation(int or tuple, optional) – 卷积核元素之间的间距
            # groups(int, optional) – 从输入通道到输出通道的阻塞连接数
            # bias(bool, optional) - 如果bias=True，添加偏置
            # 输入 3 x 96 x 96
            nn.Conv2d(3, ndf, 5, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf) x 32 x 32

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf*2) x 16 x 16

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf*4) x 8 x 8

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出 (ndf*8) x 4 x 4

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # 输出一个数(概率)
        )

    def forward(self, input):
        return self.main(input).view(-1)
