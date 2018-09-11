from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from PIL import Image

from utils.parse_config import *
from utils.utils import build_targets
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    根据module_defs（list形式）中的模块配置 来构造 网络模块list
    """
    # 第一行存放的是 超参数，所以需要pop出来
    hyperparams = module_defs.pop(0)
    # 输入图像的通道数为3
    output_filters = [int(hyperparams['channels'])]
    # 保存yolov3网络模型
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        # 解析cfg网络结构，转化为pytorch网络结构
        if module_def['type'] == 'convolutional':
            '''
            每个卷积层后都会跟一个BN层和一个LeakyReLU，算作list中的一行
            pad = 1 表示 使用pad,但是具体pad值时按照kernel_size计算的
            bn=1 也表示 使用bn,具体值为 输出通道数
            '''
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            # // 表示先做除法，然后向下取整
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                # 值为 输出通道数
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if module_def['activation'] == 'leaky':
                # 激活函数
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

        elif module_def['type'] == 'upsample':
            '''
            上采样与rount搭配使用
            
            上采样将feature map变大，然后与 之前的较大feature map在深度上合并
            '''

            # nearest 使用最邻近 nrighbours 对输入进行采样 像素值.
            upsample = nn.Upsample( scale_factor=int(module_def['stride']),
                                    mode='nearest')
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            '''
            route 指 按照列来合并tensor,即扩展深度
            filters为该层输出，保存到output_filters
            '''
            layers = [int(x) for x in module_def["layers"].split(',')]
            filters = sum([output_filters[layer_i] for layer_i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            '''
            shortcut 指  残差结构，卷积的跨层连接，即 将不同两层输出（即输出+残差块）相加 为 最后结果
            filters为该层输出，保存到output_filters
            '''
            filters = output_filters[int(module_def['from'])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())

        elif module_def["type"] == "yolo":
            '''
            对于YOLOLayer层：
            训练阶段返回 各loss
            预测阶段返回  预测结果
            '''
            # mask为 即从 anchor集合中选用哪几个anchor
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors  提取anchor
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            # 只拿 该层挑选之后的anchor
            anchors = [anchors[i] for i in anchor_idxs]
            # 数据集共多少类别。coco数据集80类别
            num_classes = int(module_def['classes'])
            # 输入的训练图像大小416
            img_height = int(hyperparams['height'])
            # Define detection layer 定义检测层
            yolo_layer = YOLOLayer(anchors, num_classes, img_height)
            modules.add_module('yolo_%d' % i, yolo_layer)
        # Register module list and number of output filters
        # 注册模块列表和输出过滤器的数量
        #保存 模型结构list
        module_list.append(modules)
        # 保存每层的输出结果list
        output_filters.append(filters)

    return hyperparams, module_list

class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""
    '''
    “route”和“shortcut”层的占位符
    '''
    def __init__(self):
        super(EmptyLayer, self).__init__()

class YOLOLayer(nn.Module):
    """Detection layer"""
    '''
    检测层
    训练时计算损失
    预测时输出预测结果
    '''
    def __init__(self, anchors, num_classes, img_dim):
        '''
        :param anchors: 该检测层 挑选的几个anchor
        :param num_classes: 数据集类别，coco数据集共80类
        :param img_dim: 输入图像大小416
        '''
        super(YOLOLayer, self).__init__()
        self.anchors = anchors    #该检测层 挑选的几个anchor
        self.num_anchors = len(anchors)
        self.num_classes = num_classes  #数据集类别，coco数据集共80类
        self.bbox_attrs = 5 + num_classes  #一个 网格需要预测的值个数
        self.img_dim = img_dim   # 输入训练图像的大小
        self.ignore_thres = 0.5  #  是否为物体的阈值（ 预测结果，即物体置信度小于该阈值，则认为该处没有预测到物体）
        self.lambda_coord = 1  #计算损失时的lambda，一般默认为1（损失公式中，用于调节 分类  和 检测  的比重）

        self.mse_loss = nn.MSELoss()   #均方误差 损失函数，计算 检测时的坐标损失
        self.bce_loss = nn.BCELoss()  #计算目标和输出之间的二进制交叉熵  损失函数，计算  多类别的分类损失

    def forward(self, x, targets=None):
        # yolo有3个检测层13x13,26x26,52x52，这里以 第一个检测层13x13为例
        # x [16,255,13,13]  16:batch数    255：深度   13x13：feature map大小
        bs = x.size(0)
        g_dim = x.size(2)  # feature map大小
        stride =  self.img_dim / g_dim   # feature相对于原图416的缩放倍数   32
        # Tensors for cuda support   设置初始化tensor的默认类型
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # [16,3,13,13,85]     contiguous返回一个内存连续的有相同数据的 tensor
        prediction = x.view(bs,  self.num_anchors, self.bbox_attrs, g_dim, g_dim).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs    85中0-3 为预测的框偏移，4为 物体置信度（是否有物体）  5： 为多类别的分类概率
        x = torch.sigmoid(prediction[..., 0])          # Center x  [16,3,13,13]
        y = torch.sigmoid(prediction[..., 1])          # Center y  [16,3,13,13]
        w = prediction[..., 2]                         # Width     [16,3,13,13]
        h = prediction[..., 3]                         # Height    [16,3,13,13]
        conf = torch.sigmoid(prediction[..., 4])       # Conf      [16,3,13,13]
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred. [16,3,13,13,80]

        # Calculate offsets for each grid 计算每个网格的偏移量
        # torch.linspace返回 start 和 end 之间等间隔 steps 点的一维 Tensor
        # repeat沿着指定的尺寸重复 tensor
        # 过程：
        #      torch.linspace(0, g_dim-1, g_dim)  ->  [1,13]的tensor
        #      repeat(g_dim,1)                    ->  [13,13]的tensor 每行内容为0-12,共13行
        #      repeat(bs*self.num_anchors, 1, 1)  ->  [48,13,13]的tensor   [13,13]内容不变，在扩展的一维上重复48次
        #      view(x.shape)                      ->  resize成[16.3.13.13]的tensor
        # grid_x、grid_y用于 定位 feature map的网格左上角坐标
        grid_x = torch.linspace(0, g_dim-1, g_dim).repeat(g_dim,1).repeat(bs*self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)    # [16.3.13.13]  每行内容为0-12,共13行
        grid_y = torch.linspace(0, g_dim-1, g_dim).repeat(g_dim,1).t().repeat(bs*self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)  # [16.3.13.13]  每列内容为0-12,共13列（因为使用转置T）
        scaled_anchors = [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]  #将 原图尺度的锚框也缩放到统一尺度下
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))  #[3,1]  3个锚框的w值
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))  #[3,1]  3个锚框的h值
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, g_dim*g_dim).view(w.shape) #[16,3,13,13]
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, g_dim*g_dim).view(h.shape) #[16,3,13,13]

        # Add offset and scale with anchors  给锚框添加偏移量和比例
        pred_boxes = FloatTensor(prediction[..., :4].shape)  #新建一个tensor[16,3,13,13,4]
        # pred_boxes为 在13x13的feature map尺度上的预测框
        # x,y为预测值（网格内的坐标，经过sigmoid之后值为0-1之间） grid_x，grid_y定位网格左上角偏移坐标（值在0-12之间）
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        # w，h为 预测值，即相对于原锚框的偏移值    anchor_w，anchor_h为 网格对应的3个锚框
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # Training 训练阶段
        if targets is not None:

            # 将损失函数转到GPU上，第一次见...
            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
            # nGT 统计一个batch中的真值框个数
            # nCorrect 统计 一个batch预测出有物体的个数
            # mask   [16,3,13,13]全0   在3个原始锚框与 真值框 iou最大的那个锚框  对应的预测框位置置为1 ，即  负责检测物体的位置为1
            # conf_mask  [16,3,13,13]  初始化全1，之后的操作：负责预测物体的网格置为1，它周围网格置为0
            # tx, ty [16,3,13,13] 初始化全为0，在有真值框的网格位置写入 真实的物体中心点坐标
            # tw, th  [16,3,13,13] 初始化全为0，该值为 真值框的w、h 按照公式转化为 网络输出时对应的真值（该值对应于 网络输出的真值）
            # tconf [16,3,13,13]   初始化全0，有真值框对应的网格位置为1  标明 物体中心点落在该网格中，该网格去负责预测物体
            # tcls    #[16,3,13,13,80]  初始化全0，经过one-hot编码后  在真实类别处值为1
            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(pred_boxes.cpu().data,   #在13x13尺度上的预测框  [16,3,13,13,4]
                                                                            targets.cpu().data,                  #坐标被归一化后的真值框filled_labels[16,50,5] 值在0-1之间
                                                                            scaled_anchors,                      #缩放到13x13尺度下的3个锚框
                                                                            self.num_anchors,                    #锚框个数3
                                                                            self.num_classes,                    #数据集类别数  coco数据集80
                                                                            g_dim,                               #feature map相对于原图的缩放倍数13
                                                                            self.ignore_thres,                   # 阈值（用于判断  真值框 与 3个原始锚框的iou > 阈值）
                                                                            self.img_dim)                        #网络输入图像的大小 416
            #  conf[16,3,13,13] 为网络输出值，物体置信度（是否有物体）
            nProposals = int((conf > 0.25).sum().item())
            # 召回率recall = 预测出有物体 / 真值框总数
            recall = float(nCorrect / nGT) if nGT else 1

            # Handle masks
            mask = Variable(mask.type(FloatTensor))
            cls_mask = Variable(mask.unsqueeze(-1).repeat(1, 1, 1, 1, self.num_classes).type(FloatTensor))
            conf_mask = Variable(conf_mask.type(FloatTensor))

            # Handle target variables
            tx    = Variable(tx.type(FloatTensor), requires_grad=False)
            ty    = Variable(ty.type(FloatTensor), requires_grad=False)
            tw    = Variable(tw.type(FloatTensor), requires_grad=False)
            th    = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls  = Variable(tcls.type(FloatTensor), requires_grad=False)

            # Mask outputs to ignore non-existing objects  通过掩码来忽略 不存在物体
            # mask 初始化全为0，只有  在3个原始锚框与 真值框 iou最大的那个锚框  对应的预测框位置置为1，即  负责检测物体的位置为1
            loss_x = self.lambda_coord * self.bce_loss(x * mask, tx * mask)
            loss_y = self.lambda_coord * self.bce_loss(y * mask, ty * mask)
            loss_w = self.lambda_coord * self.mse_loss(w * mask, tw * mask) / 2   # 为何 /2 ?
            loss_h = self.lambda_coord * self.mse_loss(h * mask, th * mask) / 2
            # 有无物体损失  conf_mask  [16,3,13,13]  初始化全1，之后的操作：负责预测物体的网格置为1，它周围网格置为0
            loss_conf = self.bce_loss(conf * conf_mask, tconf * conf_mask)
            # 多分类损失
            loss_cls = self.bce_loss(pred_cls * cls_mask, tcls * cls_mask)
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return loss, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item(), recall

        else:
            # If not in training phase return predictions
            # 预测阶段，返回 预测结果
            output = torch.cat((pred_boxes.view(bs, -1, 4) * stride, conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
            return output.data


class Darknet(nn.Module):
    """YOLOv3 object detection model"""
    '''
    YOLOv3物体检测模型
    '''
    def __init__(self, config_path, img_size=416):
        '''
        输入通常为416（32的倍数）
        理由：参与预测层的最小特征图为13x13,为原图缩小32倍
        '''
        super(Darknet, self).__init__()
        # 将cfg配置文件转化为list,每一行 为网络的一部分
        self.module_defs = parse_model_config(config_path)
        # 解析list，返回 pytorch模型结构
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        # 即训练网络过程中使用的图像总个数 （官方权重内seen值为32013312）
        self.seen = 0
        # 保存模型时文件头写入的信息（5个字符，其余可不写）
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ['x', 'y', 'w', 'h', 'conf', 'cls', 'recall']

    def forward(self, x, targets=None):
        # True: 训练阶段    False:预测阶段
        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        # 保存每一层的网络输出值
        layer_outputs = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample']:
                x = module(x)
            elif module_def['type'] == 'route':
                '''
                route 指 按照列来合并tensor,即扩展深度
                
                当属性只有一个值时，它会输出由该值索引的网络层的特征图。
                在我们的示例中，它是−4，因此这个层将从Route层开始倒数第4层的特征图。

                当图层有两个值时，它会返回由其值所索引的图层的连接特征图。 
                在我们的例子中，它是−1,61，并且该图层将输出来自上一层（-1）和第61层的特征图，并沿深度的维度连接。
                
                '''
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def['type'] == 'shortcut':
                '''
                shortcut 指  残差结构，卷积的跨层连接，即 将不同两层输出（即输出+残差块）相加 为 最后结果
                参数from是−3，意思是shortcut的输出是通过与先前的倒数第三层网络相加而得到。
                '''
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'yolo':
                # Train phase: get loss
                # 训练阶段：获得损失
                if is_training:
                    # x为总loss, *losses为各种loss
                    x, *losses = module[0](x, targets)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                # 测试阶段：获取检测
                else:
                    x = module(x)
                output.append(x)
            layer_outputs.append(x)

        self.losses['recall'] /= 3
        # 训练阶段：返回总loss 用于梯度更新
        # 预测阶段：返回  预测结果
        return sum(output) if is_training else torch.cat(output, 1)


    def load_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""
        '''
        解析并加载存储在'weights_path中的权重
        '''

        #Open the weights file
        fp = open(weights_path, "rb")
        # First five are header values  前五个为标题信息
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)

        # Needed to write header when saving weights
        # 保存权重时需要写头
        self.header_info = header

        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)         # The rest are weights
        fp.close()

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                if module_def['batch_normalize']:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel() # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    # 如果设置的是False，只需加载卷积层的偏置即可
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                # 最终，加载卷积层参数：
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w


    def save_weights(self, path, cutoff=-1):
        """
            保存模型权重(仅保存卷积层conv、BN层batch_normalize的权重参数信息，其余参数如shortcut、rount等为定值，无需保存)
            权重文件是包含以串行方式存储的权重的二进制文件
            当BN层出现在卷积块中时，不存在偏差。 但是，当没有BN layer 时，偏差“权重”必须从文件中读取

            @:param path    - path of the new weights file  （保存路径）
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
            当cutoff=-1时：保存全部网络参数
            当cutoff不为-1时，保存指定的部分网络参数
        """
        fp = open(path, 'wb')
        self.header_info[3] = self.seen
        # tofile 将数组中的数据以二进制格式写进文件。文件路径为fp
        self.header_info.tofile(fp)

        # Iterate through layers 遍历网络层
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
