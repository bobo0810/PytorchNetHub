from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F

import numpy as np
from PIL import Image
import cv2

from functools import partial
import sys
import json

import pdb

# load a pretrained model, such a model already has a global pooling at the end
# model_id: 1 - SqueezeNet, 2 - ResNet, 3 - DenseNet
def load_model(model_id):
    if model_id == 1:
        model = models.squeezenet1_1(pretrained = True)
        final_conv_layer = 'classifier.1'
    elif model_id == 2:
        model = models.resnet101(pretrained = True)
        final_conv_layer = 'layer4'
    elif model_id == 3:
        model = models.densenet161(pretrained = True)
        final_conv_layer = 'features'
    else:
        sys.exit('No such model!')

    return model, final_conv_layer

# a hook to a given layer
def hook(module, input, output, feature_blob):
    feature_blob.append(output.data.numpy())

# load and preprocess an image
def load_image(filename = './test.jpb'):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Scale((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    image = Image.open(filename)
    image = preprocess(image)

    return Variable(image.unsqueeze(0))

# read in labels, original file url: https://s3.amazonaws.com/outcome-blog/imagenet/labels.json
def get_labels(filename = '.labels.json'):
    with open(filename) as f:
        content = json.load(f)

    labels = {int(k) : v for (k, v) in content.items()}

    return labels

# compute class activation map
def compute_cam(activation, softmax_weight, class_ids):
    '''
     对于 top-5 的预测 生成CAM  类激活映射
    :param activation:  最后一个conv层输出的结果  [1,2048,7,7]
    :param softmax_weight: 全连接的权重 [1000,2048]
    :param class_ids: 按照概率从大到小排序后的 前五个概率的下标
    '''
    # b:  batch size
    # c:  channel 通道数
    b, c, h, w = activation.shape
    cams = []
    for idx in class_ids:
        #[2048,49]
        activation = activation.reshape(c, h * w)
        # dot 计算两个张量的点乘 (内积). 矩阵乘法
        # [1.49]
        cam = softmax_weight[idx].dot(activation)
        # [7.7]
        cam = cam.reshape(h, w)
        # 归一化 normalize to [0, 1]
        cam =  (cam - cam.min()) / (cam.max() - cam.min())
        # 转化到0-255之间 conver to [0, 255]
        cam = np.uint8(255 * cam)
        # reshape to (224, 224)
        # 从[7,7] resize到[224,224]
        cams.append(cv2.resize(cam, (224, 224)))

    return cams

if __name__ == '__main__':

    # if len(sys.argv) != 2:
    #     sys.exit('Wrong number of arguments!')

    # try:
    #     model_id = int(sys.argv[1])
    # except:
    #     sys.exit('Wrong second argument')

    # model_id: 1 - SqueezeNet, 2 - ResNet, 3 - DenseNet  都为预训练模型
    model_id=2  # 1不能用
    # load a pretrained model
    model, final_conv_layer = load_model(model_id)
    model.eval()

    # add a hook to a given layer
    # 为最后一个conv层添加一个钩子
    feature_blob = []
    model._modules.get(final_conv_layer).register_forward_hook(partial(hook, feature_blob = feature_blob))

    # get the softmax (last fc layer) weight
    # 得到 softmax(即 最后一层fc层)权重
    params = list(model.parameters())
    softmax_weight = np.squeeze(params[-2].data.numpy())
    #[1,3,224,224]
    input = load_image('./test.jpg')
    #[1,1000]
    output = model(input)   # scores
    #dict 1000
    labels = get_labels('./labels.json')
    # [1000]
    probs = F.softmax(output).data.squeeze()
    # 按照 概率进行从大到小排序
    probs, idx = probs.sort(0, descending = True)


    # output the top-5 prediction
    print('输出top-5的预测结果')
    for i in range(5):
        print('{:.3f} -> {}'.format(probs[i], labels[idx[i]]))

    # generate class activation map for the top-5 prediction
    # 对于 top-5 的预测 生成CAM  类激活映射
    # feature_blob[0]  为最后一个conv层输出的结果  [1,2048,7,7]
    # softmax_weight  为全连接的权重 [1000,2048]
    # idx[0: 5]  为 按照概率从大到小排序后的 前五个概率的下标
    # cams list:5  每个里面都是[224,224]
    cams = compute_cam(feature_blob[0], softmax_weight, idx[0: 5])

    for i in range(len(cams)):
        # render cam and original image
        # 将 cam类激活映射 和 原图  组合在一起
        filename = labels[idx[i]] + '.jpg'
        print('output %s for the top-%s prediction: %s' % (filename, (i + 1), labels[idx[i]]))

        # 原图 [216,380,3]
        img = cv2.imread('./test.jpg')
        h, w, _ = img.shape

        # 从cams类激活映射 转化的 热力图 [216,380,3]
        heatmap = cv2.applyColorMap(cv2.resize(cams[i], (w, h)), cv2.COLORMAP_JET)

        # 最终结果= 0.3热力图  +  0.5原图
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite(filename, result)
