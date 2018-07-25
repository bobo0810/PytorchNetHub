import torch.nn as nn
import torch.nn.init as init
'''
 使用xavier方法来初始化vgg后面的新增层、loc用于回归层、conf用于分类层  的权重
'''
def xavier(param):
    '''
    使用xavier算法初始化新增层的权重
    '''
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()