import torch.nn as nn
import numpy as np
import torch
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F



def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock(nn.Module):
    '''RepVGG模块'''
    #   kernel_size=3, stride=2, padding=1    降2倍分辨率
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2 # 计算1x1卷积的padding数
 
        self.nonlinearity = nn.ReLU() # 非线性激活函数

 
        self.se = nn.Identity() # 恒等映射，y=x

        # 部署: 单个conv
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        # 训练: 3分支
        else:
            # 恒等映射： （1） 当分辨率和通道不变时，采用 恒等映射+BN    （2）各阶段的第一个conv降2倍分辨率，无法恒等映射，改为None 无任何操作。
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            # 3x3conv + bn
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            # 1x1conv + bn
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            print('RepVGG Block, identity = ', self.rbr_identity)


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))


    def get_equivalent_kernel_bias(self):
        '''
        核心代码  转为等价的vgg结构
        '''
        # conv融合bn
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense) # kernel [输出通道，输入通道, 3,3] , bias [输出通道]
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1) # kernel [输出通道，输入通道, 1,1] , bias [输出通道]
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity) # 当分辨率减半时，无恒等映射，值为0,0。  
        # 1个卷积 等价转换为 1个卷积
        # 即conv(x, W1) + conv(x, W2) + conv(x, W3) = conv(x, W1+W2+W3)）
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        '''1x1conv 填充为等价的3x3 conv'''
        if kernel1x1 is None:
            return 0
        else:
            # https://zhuanlan.zhihu.com/p/358599463
            # pad = (左边填充数， 右边填充数， 上边填充数， 下边填充数)
            # 1x1卷积核的上下左右四边各填充1个零值，变为3x3 conv
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])
    

    def _fuse_bn_tensor(self, branch):
        '''conv+bn融合为conv'''
        if branch is None: # 恒等映射
            return 0, 0
        if isinstance(branch, nn.Sequential):  # conv_bn-> conv
            kernel = branch.conv.weight # 卷积核参数  [输出通道，输入通道,  卷积核长,卷积核宽] 
            running_mean = branch.bn.running_mean # 训练阶段统计的均值  [输出通道]
            running_var = branch.bn.running_var # 训练阶段统计的方差   [输出通道]
            gamma = branch.bn.weight # 缩放操作的γ
            beta = branch.bn.bias # 缩放操作的β
            eps = branch.bn.eps # 1e-5 防止归一化时除以0
        else: # bn -> conv
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt() # 标准差 = bn方差的开平方   形状[输出通道]
        t = (gamma / std).reshape(-1, 1, 1, 1)
        # 返回卷积核的参数 kernel 形如[输出通道，输入通道, 3,3] , bias 形如[输出通道]
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        # 出现该参数 代表已经融合为部署模型，直接返回
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True



class RepVGG(nn.Module):
    '''
    初始化网络
    '''
    # RepVGG_A0为例
    # num_blocks: 5个阶段的block数量  [2, 4, 14, 1]
    # num_classes: 类别数 
    # width_multiplier: 调整通道数的比例 [0.75, 0.75, 0.75, 2.5]
    # override_groups_map: None
    # deploy: False
    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
       

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0])) # 确定阶段0的输出通道
        # 0~4共5个阶段，每个阶段开头的特征图分辨率均降2倍。总计降32倍 
        # 阶段0：提取浅层特征
        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=self.deploy)
        
        
        self.cur_layer_idx = 1 # 当前层的序号
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)


    def _make_stage(self, planes, num_blocks, stride):
        '''构建1~4阶段的网络结构'''
        # 每个阶段内，第一个block步长为2(分辨率减半)，剩余block步长为1(分辨率不变)
        strides = [stride] + [1]*(num_blocks-1) 
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy))
            self.in_planes = planes
            self.cur_layer_idx += 1 # 层数加1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    
    
    for module in model.modules(): 
        if hasattr(module, 'switch_to_deploy'): # 判断是否存在该方法
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model

def create_RepVGG_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy)


if __name__ == '__main__':
    
    # 输入
    input=torch.ones((8,3,224,224))
    
    # 初始化模型  以RepVGG_A0为例
    model = create_RepVGG_A0(deploy=False)
    output=model(input)
  
    # 部署时转换为VGG结构
    repvgg_model_convert(model, save_path='repvgg_deploy.pth')
    print()