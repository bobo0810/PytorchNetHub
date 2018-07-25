# coding:utf8

import torch as t
import torchvision as tv
import torchnet as tnt

from torch.utils import data
from transformer_net import TransformerNet
import utils
from PackedVGG import Vgg16
from torch.autograd import Variable
from torch.nn import functional as F
import tqdm
import os
import ipdb
# ImageNet上运算的均值方差
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class Config(object):
    image_size = 256  # 图片大小
    batch_size = 8
    data_root = 'data/'  # 数据集存放路径：data/coco/a.jpg
    num_workers = 4  # 多线程加载数据
    use_gpu = True  # 使用GPU

    style_path = 'style.jpg'  # 风格图片存放路径
    lr = 1e-3  # 学习率

    env = 'neural-style'  # visdom env
    plot_every = 10  # 每10个batch可视化一次

    epoches = 2  # 训练epoch

    content_weight = 1e5  # content_loss 的权重
    style_weight = 1e10  # style_loss的权重

    model_path = None  # 预训练模型的路径
    debug_file = '/tmp/debugnn'  # touch $debug_fie 进入调试模式

    content_path = 'input.png'  # 需要进行分割迁移的图片
    result_path = 'output.png'  # 风格迁移结果的保存路径


def train(**kwargs):
    opt = Config()
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    # 可视化操作
    vis = utils.Visualizer(opt.env)

    # 数据加载
    transfroms = tv.transforms.Compose([
        # 将输入的`PIL.Image`重新改变大小成给定的`size`  `size`是最小边的边长
        tv.transforms.Scale(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        # 转为0-1之间
        tv.transforms.ToTensor(),
        # 转为0-255之间
        tv.transforms.Lambda(lambda x: x * 255)
    ])
    # 封装数据集，并进行数据转化
    dataset = tv.datasets.ImageFolder(opt.data_root, transfroms)
    # 数据加载器
    dataloader = data.DataLoader(dataset, opt.batch_size)

    # 转换网络
    transformer = TransformerNet()
    if opt.model_path:
        transformer.load_state_dict(t.load(opt.model_path, map_location=lambda _s, _: _s))

    # 损失网络 Vgg16  置为预测模式
    vgg = Vgg16().eval()

    # 优化器（需要训练 风格转化网络的参数）
    optimizer = t.optim.Adam(transformer.parameters(), opt.lr)

    # 获取风格图片的数据  形状 1*c*h*w， 分布 -2~2（使用预设）
    style = utils.get_style_data(opt.style_path)
    # 可视化风格图：-2 到2 转化为0-1
    vis.img('style', (style[0] * 0.225 + 0.45).clamp(min=0, max=1))

    if opt.use_gpu:
        transformer.cuda()
        style = style.cuda()
        vgg.cuda()

    # 风格图片的gram矩阵
    style_v = Variable(style, volatile=True)
    # 得到vgg中间四层的结果（用以跟输入图片的输出四层比较，计算损失）
    features_style = vgg(style_v)
    # gram_matrix：输入 b,c,h,w  输出 b,c,c 计算gram矩阵（四层的gram矩阵）
    gram_style = [Variable(utils.gram_matrix(y.data)) for y in features_style]

    # 损失统计  仪表盘 用以可视化（每个epoch中的所有batch平均损失）
    # 风格损失
    style_meter = tnt.meter.AverageValueMeter()
    # 内容损失
    content_meter = tnt.meter.AverageValueMeter()

    for epoch in range(opt.epoches):
        # 仪表盘清零
        content_meter.reset()
        style_meter.reset()

        for ii, (x, _) in tqdm.tqdm(enumerate(dataloader)):

            # 训练
            optimizer.zero_grad()
            if opt.use_gpu:
                x = x.cuda()
            # x为输入的真实图像
            x = Variable(x)
            # 风格转换后的预测图像为y
            y = transformer(x)
            # 输入: b, ch, h, w   0~255
            # 输出: b, ch, h, w    - 2~2
            # 将x,y范围从0-255转化为-2-2
            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)
            # 返回 四个中间层的特征输出
            features_y = vgg(y)
            features_x = vgg(x)

            # content loss内容损失 只计算relu2_2之间的损失   预测图片与原图在relu2_2中间层比较，计算损失
            # content_weight内容的权重     mse_loss均方误差损失函数
            content_loss = opt.content_weight * F.mse_loss(features_y.relu2_2, features_x.relu2_2)

            # style loss
            style_loss = 0.
            # 风格损失取四层的均方误差损失总和
            # features_y：预测图像的四层输出内容    gram_style：风格图像的四层输出的gram_matrix
            # zip将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
            for ft_y, gm_s in zip(features_y, gram_style):
                # 计算预测图像的四层输出内容的gram_matrix
                gram_y = utils.gram_matrix(ft_y)
                style_loss += F.mse_loss(gram_y, gm_s.expand_as(gram_y))
            style_loss *= opt.style_weight
            # 总损失=风格损失+内容损失
            total_loss = content_loss + style_loss
            # 反向传播
            total_loss.backward()
            # 更新参数
            optimizer.step()

            # 损失平滑  将损失加入仪表盘，以便可视化损失过程
            content_meter.add(content_loss.data[0])
            style_meter.add(style_loss.data[0])
            # 每plot_every次前向传播后可视化
            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # 可视化
                vis.plot('content_loss', content_meter.value()[0])
                vis.plot('style_loss', style_meter.value()[0])
                # 因为x和y经过标准化处理(utils.normalize_batch)，所以需要将它们还原
                #x,y为[-2,2]还原回[0,1]
                vis.img('output', (y.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))
                vis.img('input', (x.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))

        # 每次epoch完毕后保存visdom和模型
        vis.save([opt.env])
        t.save(transformer.state_dict(), 'checkpoints/%s_style.pth' % epoch)

# 生成图片  预测
def stylize(**kwargs):
    opt = Config()

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    # 图片处理
    content_image = tv.datasets.folder.default_loader(opt.content_path)
    content_transform = tv.transforms.Compose([
        # 转为[0,1]
        tv.transforms.ToTensor(),
        # 转为[0,255]
        tv.transforms.Lambda(lambda x: x.mul(255))
    ])
    # 图片转化
    content_image = content_transform(content_image)
    # 扩充第0维
    content_image = content_image.unsqueeze(0)
    content_image = Variable(content_image, volatile=True)

    # 模型
    style_model = TransformerNet().eval()
    style_model.load_state_dict(t.load(opt.model_path, map_location=lambda _s, _: _s))

    if opt.use_gpu:
        content_image = content_image.cuda()
        style_model.cuda()

    # 风格迁移与保存    通过风格转化网络预测图片
    output = style_model(content_image)
    output_data = output.cpu().data[0]
    # 转化为0-1 保存图像
    tv.utils.save_image(((output_data / 255)).clamp(min=0, max=1), opt.result_path)


if __name__ == '__main__':
    import fire

    fire.Fire()
