from collections import namedtuple
import time
from torch.nn import functional as F
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator

from torch import nn
import torch as t
from torch.autograd import Variable
from utils import array_tool as at
from utils.vis_tool import Visualizer

from utils.config import opt
from torchnet.meter import ConfusionMeter, AverageValueMeter
#Tuple元祖。对应的loss名称
LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])


class FasterRCNNTrainer(nn.Module):
    """wrapper for conveniently training. return losses
       wrapper以便方便训练，返回losses
    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for  Region Proposal Network (RPN).
                           RPN定位loss
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
                           RPN分类loss
    * :obj:`roi_loc_loss`: The localization loss for the head module.
                            roi定位loss
    * :obj:`roi_cls_loss`: The classification loss for the head module.
                            roi分类loss
    * :obj:`total_loss`: The sum of 4 loss above.
                          4个loss之和

    Args:
        faster_rcnn (model.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
    """

    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()
        #传入的是FasterRCNNVGG16模型，继承了FasterRCNN模型，而参数根据说明 是FasterRCNN模型
        #即初始化的是FasterRCNN模型
        #FasterRCNN模型是父类   FasterRCNNVGG16模型是子类
        self.faster_rcnn = faster_rcnn
        #sigma for l1_smooth_loss
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # target creator create gt_bbox gt_label etc as training targets.
        #目标框creator 目标是产生 真实的bbox 类别标签等
        #将真实的bbox分配给锚点
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()
        #得到faster网络权重，均值 和方差
        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        #得到faster网络的优化器
        self.optimizer = self.faster_rcnn.get_optimizer()
        # visdom wrapper
        self.vis = Visualizer(env=opt.env)

        # indicators for training status
        #训练状态指标  两个混淆矩阵 2×2（前景后景）   21×21（20类+背景）
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss 平均损失

    def forward(self, imgs, bboxes, labels, scale):
        """Forward Faster R-CNN and calculate losses.
        Faster网络的前向传播、计算losses*************************
        Here are notations used.

        * :math:`N` is the batch size. `N`是批量大小
        * :math:`R` is the number of bounding boxes per image. `R`是每个图像的边界框的数量

        Currently, only :math:`N=1` is supported.
        当前模型，只有N=1可用

        Args:
            imgs (~torch.autograd.Variable): A variable with a batch of images.
                                            batch=1的图片变量
            bboxes (~torch.autograd.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
                                            真实人工标注的bboxes变量
            labels (~torch.autograd..Variable): A batch of labels.
                Its shape is :math:`(N, R)`.
                 The background is excluded from the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground classes.
                 背景被排除在定义之外，这意味着值的范围。`L`是前景类的数量
            scale (float): Amount of scaling applied to
                the raw image during preprocessing.
                预处理期间应用于原始图像的缩放量

        Returns:
            namedtuple of 5 losses
            五个损失
        """

        n = bboxes.shape[0]
        #判断，只支持batch为1
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')
        #img_size=原图像的高、宽
        _, _, H, W = imgs.shape
        img_size = (H, W)
        #通过提取器（预训练好的VGG16）网络提取特征
        features = self.faster_rcnn.extractor(imgs)
        #通过rpn网络（区域提案网络）得到
        #rpn这是一个区域提案网络。它提取图像特征，预测输出rois
        #rpn_locs[1,17316,4]   rpn_scores[1,17316,2]   rois[2000,4]   roi_indices[2000,]全为0  anchor [17316,4]
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.faster_rcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        # 由于批量大小为1，因此将变量转换为单数形式（即压缩第一维）
        #bbox变为[1,4]
        bbox = bboxes[0]
        label = labels[0]
        #则rpn_score变为[17316,4]  rpn_loc 变为[17316,2]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        #大约2000个rois
        roi = rois

        # Sample RoIs and forward   简单的ROIs和前向传播
        # it's fine to break the computation graph of rois, consider them as constant input
        #打破rois的计算图，将它作为一个固定不变的输入
        #proposal_target_creator  输入为rois（2000个候选框，和人工标注的bbox）用于生成训练目标，只训练用到
        #2000个rois选出128个
        #sample_roi[128,4]     gt_roi_loc[128,4]     gt_roi_label[128,] 值为0或1 表示正负样本
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        # NOTE it's all zero because now it only support for batch=1 now
        #它全部为零，因为现在它只支持batch = 1
        sample_roi_index = t.zeros(len(sample_roi))
        #roi head网络进行预测类别和目标框
        #RoIHead： 负责对rois分类和微调。对RPN找出的rois，判断它是否包含目标，并修正框的位置和座标
        #使用RoIs提议的的feature maps，对RoI中的对象进行分类并提高目标框定位
        #roi_cls_loc  roi的分类、回归
        #传入  特征提取的features   和  128个ROI
        #roi_cls_loc [128,84]回归定位    roi_score[128,21]分类（20类加背景）
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)

        # ------------------ RPN losses -------------------#
        #真实标注的bbox,预测出来的anchor锚点
        # 将真实的bbox分配给锚点，返回 经过rpn后对应的定位和标签
        #gt_rpn_loc[17316,4]     gt_rpn_label  [17316,]
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            img_size)
        #转为变量V  转为long型
        gt_rpn_label = at.tovariable(gt_rpn_label).long()
        gt_rpn_loc = at.tovariable(gt_rpn_loc)
        #rpn的回归定位损失   rpn_loc_loss[1]
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...
        #ignore_index的默认值是 - 100...
        #F：pytorch的function
        #分类使用交叉熵
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)

        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
        #添加进rpn 混淆矩阵
        self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        #roi分类和回归   压缩第一维
        #n_sample 128
        n_sample = roi_cls_loc.shape[0]
        #改变形状为[ 32,4]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        #得到roi的回归
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), \
                              at.totensor(gt_roi_label).long()]
        # gt_roi_label：真实roi的标签
        #gt_roi_loc：真实roi的回归
        gt_roi_label = at.tovariable(gt_roi_label).long()
        gt_roi_loc = at.tovariable(gt_roi_loc)
        #roi的回归损失  计算回归定位的损失
        roi_loc_loss = _fast_rcnn_loc_loss(
            #contiguous从不连续调整为连续
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)
        #roi分类损失（交叉熵）
        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())
        #添加进roi 混淆矩阵
        self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())
        #计算总损失
        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]
        #返回Tuple，四个损失+总损失
        return LossTuple(*losses)
    #训练并更新可学习参数
    def train_step(self, imgs, bboxes, labels, scale):
        #优化器梯度清零
        self.optimizer.zero_grad()
        #前向传播（重点*）  返回（总损失 和四类损失）
        losses = self.forward(imgs, bboxes, labels, scale)
        #反向传播（重点*）
        #针对总损失进行反向传播
        losses.total_loss.backward()
        # 更新可学习参数
        self.optimizer.step()
        #将losses写入meter中
        self.update_meters(losses)
        return losses

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.
        
        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        t.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self
    #更新仪表盘  用以显示
    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])
    #将值重置到0
    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        #将两个混淆矩阵的内容也置为0
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}

#计算smooth_l1损失
def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    flag = Variable(flag)
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()

#计算回归定位的损失
def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    #定位损失 只计算 正样本（存在物体的前景）的损失
    #初始化全0的tensor,将正样本对应的位置值为1。以便只计算正样本的损失
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    #使用smooth_l1（回归位置使用smooth_l1）计算损失
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, Variable(in_weight), sigma)
    # Normalize by total number of negtive and positive rois.
    #为了便于训练，对选择出的128个RoIs，还对他们的gt_roi_loc 进行标准化处理（减去均值除以标准差）
    loc_loss /= (gt_label >= 0).sum()  # ignore gt_label==-1 for rpn_loss
    return loc_loss
