from torch.cuda.amp import autocast
import torch.nn as nn
class MyNet(nn.Module):
    '''
    自定义网络
    '''
    def __init__(self, use_amp=False):
        '''
        :param use_amp: True开启混合精度训练
        '''
        super(MyNet, self).__init__()
        self.use_amp = use_amp

    def forward(self,input):
        if self.use_amp:
            # 开启自动混合精度
            with autocast():
                return self.forward_calculation(input)
        else:
            return self.forward_calculation(input)

    def forward_calculation(self, input):
        ...
        ...
        return feature