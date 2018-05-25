import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch.nn.functional as F
from  torchvision.models  import resnet152

class resnet152_bo(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(resnet152_bo, self).__init__()
        model=resnet152()
        # 先不修改网络结构，先试试只修改最后一个输出层参数
        # #取掉model的后两层(去掉最后的最大池化层和全连接层)
        self.features=nn.Sequential(*list(model.children())[:-2])
        self.classifier=nn.Sequential(
            nn.Linear(2048 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 取消一层全连接层
            # nn.Linear(4096, 4096),
            # nn.ReLU(True),
            # nn.Dropout(),
            # 最后一层修改为1470   即为1470代表一张图的信息（1470=7x7x30）
            nn.Linear(4096, 1470),
        )
        # model.fc = nn.Linear(2048, 1470)
        # self.resnet152_bo=model
        # 只修改了线形层  所以只给线形层初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            # 只修改了线形层  所以只给线形层初始化权重
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # 得到输出，经过sigmoid 归一化到0-1之间
        x = F.sigmoid(x)
        # 再改变形状，返回（xxx,7,7,30）  xxx代表几张照片，（7,7,30）代表一张照片的信息
        x = x.view(-1,7,7,30)
        return x



def test():
    '''
    测试使用
    '''
    import torch
    from torch.autograd import Variable

    model = resnet152_bo(resnet152(pretrained=True))
    img = torch.rand(2,3,224,224)
    img = Variable(img)
    output = model(img)
    output = output.view(-1, 7, 7, 30)
    print(output.size())

if __name__ == '__main__':
    test()