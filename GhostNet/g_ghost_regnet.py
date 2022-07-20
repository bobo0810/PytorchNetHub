# 2022.07.17-Changed for building GhostNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
Creates a G-Ghost RegNet Model as defined in paper:
GhostNets on Heterogeneous Devices via Cheap Operations.
https://arxiv.org/abs/2201.03297
Modified from https://github.com/d-li14/regnet.pytorch
"""
import numpy as np
import torch
import torch.nn as nn

__all__ = ['regnetx_032']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution  作用1:降尺度   作用2:改变通道数   作用3:平滑层,分组卷积后融合各通道特征"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    """每个stage内部存在多个残差瓶颈层
    1x1conv增加通道，尺度不变  低计算量扩展通道数
    3x3conv通道不变，尺度减半  低计算量的分组卷积，提取深层特征
    1x1conv通道不变，尺度不变  平滑层,分组卷积后融合各通道特征
    """
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, group_width=1,
                 dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = planes * self.expansion
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, width // min(width, group_width), dilation) #分组卷积 减少计算量
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # 1x1conv增加通道，尺度不变  低计算量扩展通道数
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3x3conv通道不变，尺度减半  低计算量的分组卷积，提取深层特征
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1x1conv通道不变，尺度不变  平滑层,分组卷积后融合各通道特征
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        # 经典的残差结构
        out += identity
        out = self.relu(out)

        return out


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
       
    
class Stage(nn.Module):
    """G-Ghost Stage with mix

    Args:
        nn (_type_): _description_
    """
    def __init__(self, block, inplanes, planes, group_width, blocks, stride=1, dilate=False, cheap_ratio=0.5):
        super(Stage, self).__init__()
        norm_layer = nn.BatchNorm2d
        downsample = None
        self.dilation = 1
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes),
            )
            
        self.base = block(inplanes, planes, stride, downsample, group_width,
                            previous_dilation, norm_layer)
        self.end = block(planes, planes, group_width=group_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer)
        
        group_width = int(group_width * 0.75)
        raw_planes = int(planes * (1 - cheap_ratio) / group_width) * group_width
        cheap_planes = planes - raw_planes
        self.cheap_planes = cheap_planes
        self.raw_planes = raw_planes
        
        # mix版G-Ghost   结构类似通道注意力SE，为节省计算量用1x1conv替代全连接层
        self.merge = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # 全局平均池化
            nn.Conv2d(planes+raw_planes*(blocks-2), cheap_planes,
                      kernel_size=1, stride=1, bias=False), 
            nn.BatchNorm2d(cheap_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(cheap_planes, cheap_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(cheap_planes),
#             nn.ReLU(inplace=True),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(cheap_planes, cheap_planes,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(cheap_planes),
#             nn.ReLU(inplace=True),
        )
        self.cheap_relu = nn.ReLU(inplace=True)
        
        layers = []
        downsample = nn.Sequential(
            LambdaLayer(lambda x: x[:, :raw_planes])
        )

        layers = []
        layers.append(block(raw_planes, raw_planes, 1, downsample, group_width,
                            self.dilation, norm_layer))
        inplanes = raw_planes
        for _ in range(2, blocks-1):
            layers.append(block(inplanes, raw_planes, group_width=group_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer))

        self.layers = nn.Sequential(*layers)
        
    
    def forward(self, input):
        x0 = self.base(input) # x0拆分为e本质特征、c冗余特征
        
        m_list = [x0] 
        e = x0[:,:self.raw_planes]   # 本质特征     
        for l in self.layers:
            e = l(e)
            m_list.append(e)
        m = torch.cat(m_list,1)
        m = self.merge(m)
        
        c = x0[:,self.raw_planes:] # 冗余特征
        c = self.cheap_relu(self.cheap(c)+m)
        
        x = torch.cat((e,c),1)
        x = self.end(x)
        return x


class RegNet(nn.Module):

    def __init__(self, block, layers, widths, num_classes=1000, zero_init_residual=True,
                 group_width=1, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(RegNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False, False]
        if len(replace_stride_with_dilation) != 4:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
        self.group_width = group_width 
        # 阶段0: 图像提取为底层特征, 尺寸减半
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # 构建阶段1
        self.layer1 = self._make_layer(block, widths[0], layers[0], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        
        self.inplanes = widths[0]
        if layers[1] > 2:
            self.layer2 = Stage(block, self.inplanes, widths[1], group_width, layers[1], stride=2,
                          dilate=replace_stride_with_dilation[1], cheap_ratio=0.5) 
        else:      
            self.layer2 = self._make_layer(block, widths[1], layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[1])
        
        self.inplanes = widths[1]
        self.layer3 = Stage(block, self.inplanes, widths[2], group_width, layers[2], stride=2,
                      dilate=replace_stride_with_dilation[2], cheap_ratio=0.5)
        
        self.inplanes = widths[2]
        if layers[3] > 2:
            self.layer4 = Stage(block, self.inplanes, widths[3], group_width, layers[3], stride=2,
                          dilate=replace_stride_with_dilation[3], cheap_ratio=0.5) 
        else:
            self.layer4 = self._make_layer(block, widths[3], layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(widths[-1] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes: # 情况1：stride=2,则需要降特征尺度   情况2：输出通道不等于输出通道,则需要改变通道数
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )
        # stage内第1个block尺度减半，通道增加
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.group_width,
                            previous_dilation, norm_layer))
        # 其他block 尺度不变，提取深层特征
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, group_width=self.group_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def regnetx_032(**kwargs):
    """初始化模型  """
    return RegNet(Bottleneck, [2, 6, 15, 2], [96, 192, 432, 1008], group_width=48, **kwargs)

if __name__ == '__main__':
    input=torch.ones((2, 3,224,224))
    model=regnetx_032()

    output=model(input)
    print(model)
    print(output.shape)