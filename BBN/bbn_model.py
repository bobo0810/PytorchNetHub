import torch
import torch.nn as nn
import torch.nn.functional as F
import math

model_urls = {
    "resnet18": "https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth",
    "resnet34": "https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth",
    "resnet50": "https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth",
    "resnet101": "https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth",
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, padding=1, bias=False, stride=stride
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, bias=False, stride=1
        )
        self.bn2 = nn.BatchNorm2d(planes)
        # self.downsample = downsample
        if stride != 1 or self.expansion * planes != inplanes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleNeck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(True)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        if stride != 1 or self.expansion * planes != inplanes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        else:
            self.downsample = None
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))

        out = self.relu2(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))

        if self.downsample != None:
            residual = self.downsample(x)
        else:
            residual = x
        out = out + residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        cfg,
        block_type,
        num_blocks,
        last_layer_stride=2,
    ):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.block = block_type
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(num_blocks[0], 64)
        self.layer2 = self._make_layer(num_blocks[1], 128, stride=2)
        self.layer3 = self._make_layer(num_blocks[2], 256, stride=2)
        self.layer4 = self._make_layer(
            num_blocks[3],
            512,
            stride=last_layer_stride,
        )

    def load_model(self, pretrain):
        print("Loading Backbone pretrain model from {}......".format(pretrain))
        model_dict = self.state_dict()
        pretrain_dict = torch.load(pretrain)
        pretrain_dict = (
            pretrain_dict["state_dict"]
            if "state_dict" in pretrain_dict
            else pretrain_dict
        )
        from collections import OrderedDict

        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                k = k[7:]
            if "fc" not in k and "classifier" not in k:
                k = k.replace("backbone.", "")
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("Backbone model has been loaded......")

    def _make_layer(self, num_block, planes, stride=1):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for now_stride in strides:
            layers.append(self.block(self.inplanes, planes, stride=now_stride))
            self.inplanes = planes * self.block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out


class BBN_ResNet(nn.Module):
    def __init__(
        self,
        cfg,
        block_type,
        num_blocks,
        last_layer_stride=2,
    ):
        super(BBN_ResNet, self).__init__()
        self.inplanes = 64
        self.block = block_type

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(num_blocks[0], 64)
        self.layer2 = self._make_layer(num_blocks[1], 128, stride=2)
        self.layer3 = self._make_layer(num_blocks[2], 256, stride=2)
        self.layer4 = self._make_layer(num_blocks[3] - 1, 512, stride=last_layer_stride)

        self.cb_block = self.block(self.inplanes, self.inplanes // 4, stride=1)
        self.rb_block = self.block(self.inplanes, self.inplanes // 4, stride=1)

    def load_model(self, pretrain):
        print("Loading Backbone pretrain model from {}......".format(pretrain))
        model_dict = self.state_dict()
        pretrain_dict = torch.load(pretrain)
        pretrain_dict = (
            pretrain_dict["state_dict"]
            if "state_dict" in pretrain_dict
            else pretrain_dict
        )
        from collections import OrderedDict

        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                k = k[7:]
            if "fc" not in k and "classifier" not in k:
                k = k.replace("backbone.", "")
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict, strict=False)
        print("Backbone model has been loaded......")

    def _make_layer(self, num_block, planes, stride=1):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for now_stride in strides:
            layers.append(self.block(self.inplanes, planes, stride=now_stride))
            self.inplanes = planes * self.block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if "feature_cb" in kwargs:
            out = self.cb_block(out)
            return out
        elif "feature_rb" in kwargs:
            out = self.rb_block(out)
            return out
        out1 = self.cb_block(out)
        out2 = self.rb_block(out)
        out = torch.cat((out1, out2), dim=1)

        return out


def res50(
    cfg,
    pretrain=True,
    pretrained_model="/data/Data/pretrain_models/resnet50-19c8e357.pth",
    last_layer_stride=2,
):
    resnet = ResNet(
        cfg,
        BottleNeck,
        [3, 4, 6, 3],
        last_layer_stride=last_layer_stride,
    )
    if pretrain and pretrained_model != "":
        resnet.load_model(pretrain=pretrained_model)
    else:
        print("Choose to train from scratch")
    return resnet


def bbn_res50(
    cfg,
    pretrain=True,
    pretrained_model="/data/Data/pretrain_models/resnet50-19c8e357.pth",
    last_layer_stride=2,
):
    resnet = BBN_ResNet(
        cfg,
        BottleNeck,
        [3, 4, 6, 3],
        last_layer_stride=last_layer_stride,
    )
    if pretrain and pretrained_model != "":
        resnet.load_model(pretrain=pretrained_model)
    else:
        print("Choose to train from scratch")
    return resnet


class BBN_Network(nn.Module):
    """
    初始化BBN网络
    """

    def __init__(self, num_classes, model, pretrained):
        super(BBN_Network, self).__init__()
        self.backbone = eval(model)(
            cfg=None,
            pretrain=True,
            pretrained_model=pretrained,
            last_layer_stride=2,
        )
        self.module = nn.AdaptiveAvgPool2d((1, 1))  # GAP全局平均池化
        self.classifier = nn.Linear(2048 * 2, num_classes, bias=True)  # bbn 故×2

    # def forward(self, x, **kwargs):
    #     if "feature_flag" in kwargs or "feature_cb" in kwargs or "feature_rb" in kwargs:
    #         return self.extract_feature(x, **kwargs)
    #     elif "classifier_flag" in kwargs:
    #         return self.classifier(x)

    #     x = self.backbone(x)
    #     x = self.module(x)
    #     x = x.view(x.shape[0], -1)

    #     x = self.classifier(x)
    #     return x

    def forward(self, params):

        if isinstance(params, dict):

            imgs1 = params["imgs1"]
            imgs2 = params["imgs2"]
            l = params["l"]
            feature_a = self.extract_feature(imgs1, feature_cb=True)
            feature_b = self.extract_feature(imgs2, feature_rb=True)
            mixed_feature = 2 * torch.cat((l * feature_a, (1 - l) * feature_b), dim=1)

            output = self.classifier(mixed_feature)
            return output
        else:
            x = params
            x = self.backbone(x)
            x = self.module(x)
            x = x.view(x.shape[0], -1)
            x = self.classifier(x)
            return x

    def extract_feature(self, x, **kwargs):
        x = self.backbone(x, **kwargs)
        x = self.module(x)  # 全局池化
        x = x.view(x.shape[0], -1)
        return x



def BBN_ResNet50(num_classes):
    model = "bbn_res50"
    pretrained = "/root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth"
    return BBN_Network(num_classes, model, pretrained)
