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
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        # 每个网格的预测框数目 （4 or 6）
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']  #值为[38, 19, 10, 5, 3, 1]  即feature map的大小
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        # mean是保存预测框的列表
        mean = []
        # 遍历不同大小的feature map
        for k, f in enumerate(self.feature_maps):
            # 遍历一个feature map上的每一个网格
            # product用于求多个可迭代对象的笛卡尔积。repeat用于指定重复生成序列的次数  参考：http://funhacks.net/2017/02/13/itertools/
            # 即若f为2，则i,j取值为00,01,10,11。即遍历每一个可能
            for i, j in product(range(f), repeat=2):
                # |fk| 是第 k 个 feature map 的大小
                f_k = self.image_size / self.steps[k]
                # 单位中心unit center x,y
                # 每一个 预测框 default box 的中心，设置为：(i+0.5|fk|,j+0.5|fk|)，其中，|fk| 是第 k 个 feature map 的大小，同时，i,j∈[0,|fk|)
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # 长宽比aspect_ratio: 1
                # 真实大小rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                # 由于长宽比为1，则w=s_k  h=s_k
                mean += [cx, cy, s_k, s_k]

                # 对于 aspect ratio 为 1 时，还增加了一个 default box长宽比aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                # 由于长宽比为1，则w=s_k_prime  h=s_k_prime
                mean += [cx, cy, s_k_prime, s_k_prime]

                # 其余的长宽比rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        # 将mean列表转化为tensor
        output = torch.Tensor(mean).view(-1, 4)
        # clip:True 将输入input张量每个元素的夹紧到区间 [min,max]，并返回结果到一个新张量
        # 操作为  如果元素>max，则置为max。min类似
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
