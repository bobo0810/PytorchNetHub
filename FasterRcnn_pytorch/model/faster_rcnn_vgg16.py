import torch as t
from torch import nn
from torchvision.models import vgg16
from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from model.roi_module import RoIPooling2D
from utils import array_tool as at
from utils.config import opt


def decom_vgg16():
    #构建模型
    # the 30th layer of features is relu of conv5_3
    #第30层功能是conv5_3的relu
    
    #如果预训练的caffe模型存在，则加载caffe模型
    if opt.caffe_pretrain:
        #vgg16方法是pytorch方法
        model = vgg16(pretrained=False)
        if not opt.load_path:
            model.load_state_dict(t.load(opt.caffe_pretrain_path))
    #否则加载pytorch自身的预训练好的vgg16模型
    else:
        model = vgg16(not opt.load_path)
    
    #第30层功能是conv5_3的relu
    #取出第30层的输出，即conv5的输出。（在pooling之前的一层取出数据作为features）
    features = list(model.features)[:30]
    #vgg16的最后三层分类层
    #init初始化方法通过classifier命名包含了最后三层分类层（全连接+激活函数）
    classifier = model.classifier
    #将三层分类层放入list中
    classifier = list(classifier)
    #删除最后一个分类层的激活函数
    del classifier[6]
    #use dropout in RoIHead(config.py)
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    #冻结前四层卷积层
    #为了节省显存，前四层卷积层的学习率设为0
    for layer in features[:10]:
        for p in layer.parameters():
            #false即为不需要梯度求导，不更新参数
            p.requires_grad = False
    #Sequential容器，模块将按照传递的顺序添加到模块中
    #feature:Variable,已冻结前四层卷积层.值为从开始到conv5_3
    #classifier：vgg16三层分类层，list放入Sequential中
    return nn.Sequential(*features), classifier


class FasterRCNNVGG16(FasterRCNN):

    """Faster R-CNN based on VGG-16.
     Faster R-CNN基于VGG-16
    FasterRCNNVGG16继承了FasterRCNN
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """
    #在vgg16中下采样16x输出conv5（pooling的前一层）
    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
        #decom_vgg16
        #extractor：特征提取器。Variable,已冻结前四层卷积层.值为从开始到conv5_3
        #classifier：分类器。vgg16最后一层的输出
        extractor, classifier = decom_vgg16()
        #新建一个rpn 区域提案网络
        #512:in_channels输入的通道大小
        #512:mid_channels中间张量的通道大小
        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )
        #新建一个roihead
        #方法在本页
        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )
       
        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
        )


class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
       Faster R-CNN Head基于VGG-16的实现
      这个class被用作head for Faster R-CNN.
       根据给定的RoI中的特征  映射输出 分类定位和分类
    
    Args:
        n_class (int): The number of classes possibly including the background.
                 可能包含背景的类的数量
        roi_size (int): Height and width of the feature maps after RoI-pooling.
                  RoI-pooling之后的 feature maps的高和宽
        spatial_scale (float): Scale of the roi is resized.
                  调整后的roi的
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = t.autograd.Variable(xy_indices_and_rois.contiguous())

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    权重初始化  截断normal和随机normal
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
