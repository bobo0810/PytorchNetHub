from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    对于每个feature map，生成预测框（中心坐标及偏移量）
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        # 300
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        # 每个网格的预测框数目 （4 or 6）
        self.num_priors = len(cfg['aspect_ratios'])
        #方差
        self.variance = cfg['variance'] or [0.1]
        # 值为[38, 19, 10, 5, 3, 1]  即feature map的尺寸大小
        self.feature_maps = cfg['feature_maps']
        #  s_k 表示先验框大小相对于图片的比例，而 s_{min} 和 s_{max} 表示比例的最小值与最大值
        # min_sizes和max_sizes用来计算s_k,s_k_prime,以便计算 长宽比为1时的两个w.h
        # 各个特征图的先验框尺度 [30, 60, 111, 162, 213, 264]
        self.min_sizes = cfg['min_sizes']
        # [60, 111, 162, 213, 264, 315]
        self.max_sizes = cfg['max_sizes']
        # 感受野大小，即相对于原图的缩小倍数
        self.steps = cfg['steps']
        # 纵横比[[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.aspect_ratios = cfg['aspect_ratios']
        # True
        self.clip = cfg['clip']
        # VOC
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        # mean 是保存预测框的列表
        mean = []
        # 遍历不同feature map的尺寸大小
        for k, f in enumerate(self.feature_maps):
            # product用于求多个可迭代对象的笛卡尔积，它跟嵌套的 for 循环等价
            # repeat用于指定重复生成序列的次数。
            # 参考：http://funhacks.net/2017/02/13/itertools/
            # 即若f为2，则i,j取值为00,01,10,11。即遍历每一个可能

            # 当k=0,f=38时，range(f)的值为（0,1，...,37）则product(range(f), repeat=2)的所有取值为（0,0）（0,1）...直到（37,0）,,,（37,37）
            # 遍历一个feature map上的每一个网格
            for i, j in product(range(f), repeat=2):
                # fk 是第 k 个 feature map 的大小
                #image_size=300  steps为每层feature maps的感受野
                f_k = self.image_size / self.steps[k]
                # 单位中心unit center x,y
                # 每一个网格的中心，设置为：(i+0.5|fk|,j+0.5|fk|)，其中，|fk| 是第 k 个 feature map 的大小，同时，i,j∈[0,|fk|)
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k


                # 总体上：先添加长宽比为1的两个w、h（比较特殊），再通过循环添加其他长宽比
                # 长宽比aspect_ratio: 1
                # 真实大小rel size: min_size
                # 先验框大小相对于图片的比例
                #计算s_k 是为了求解w、h
                s_k = self.min_sizes[k]/self.image_size
                # 由于长宽比为1，则w=s_k  h=s_k
                mean += [cx, cy, s_k, s_k]

                # 对于 aspect ratio 为 1 时，还增加了一个 default box长宽比aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                # 由于长宽比为1，则w=s_k_prime  h=s_k_prime
                mean += [cx, cy, s_k_prime, s_k_prime]

                # 其余的长宽比
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # 将mean的list转化为tensor
        output = torch.Tensor(mean).view(-1, 4)

        # clip:True 将输入input张量每个元素的夹紧到区间 [min,max]，并返回结果到一个新张量
        # 操作为  如果元素>max，则置为max。min类似
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
