from pytorchSSD.data import *
from pytorchSSD.utils.augmentations import SSDAugmentation
from pytorchSSD.layers.modules import MultiBoxLoss
from pytorchSSD.ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import visdom

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
# 设定哪种数据集
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
# 数据集根目录
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
# 基础网络，即特征提取网络（去掉全连接的预训练模型vgg16）
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
# 预训练好的模型根目录，以便加载
parser.add_argument('--resume', default='/home/bobo/windowsPycharmProject/pytorchSSD/weights/ssd300_COCO_95000.pth', type=str,
                    help='Checkpoint state_dict file to resume training from')
# 从第几个item开始
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
# 加载数据集的线程数
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
# 是否使用cuda训练
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
# 初始化学习率
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
# 动量
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
# 随机梯度下降SGD的权重衰减
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
# 学习率调整参数
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
# 默认打开可视化
parser.add_argument('--visdom', default=True, type=str2bool,
                    help='Use visdom for loss visualization')
# 保存模型的目录
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
# 解析以上参数
args = parser.parse_args()

#是否使用cuda
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    # 判断数据集是否正确
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    # 初始化ssd模型
    #  cfg['min_dim']：300×300图像
    # cfg['num_classes']  21 类别数（20类+1背景）
    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net
    #多GPU并行工作
    # 数据并行是当我们将小批量样品分成多个较小的批量批次，并且对每个较小的小批量并行运行计算。
    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        # 在程序刚开始加这条语句可以提升一点训练速度，没什么额外开销
        cudnn.benchmark = True
    # 是否加载预训练好的的SSD模型
    if args.resume:
        print('加载已训练好的SSD模型，继续训练Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        # 从头开始训练
        # 先加载去掉全连接层的vgg网络。权重为预训练好的 去掉全连接层的vgg网络权重
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('正在加载vgg网络Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('从头开始训练，初始化权重Initializing weights...')
        # initialize newly added layers' weights with xavier method
        # 使用xavier方法来初始化新增层的权重
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)
    # 使用随机梯度下降SGD
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    # SSD的损失函数
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
    # 将网络置为训练模式
    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('加载数据集Loading the dataset...')
    # 一个epoch需要几次batch  //除法操作
    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0
    # 设置可视化
    if args.visdom:
        # 标题
        vis_title = 'SSD.PyTorch on ' + dataset.name
        # 说明
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        # 每次迭代的图形
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        # 每个epoch的图形
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)
    # 数据加载器
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    # 开始迭代数据集
    batch_iterator = iter(data_loader)
    # iteration数为每个batch迭代数
    for iteration in range(args.start_iter, cfg['max_iter']):
        # create batch iterator
        # 可视化
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, 'append',
                            'append', epoch_size)
            # reset epoch loss counters
            # 重置epoch损失计数器
            loc_loss = 0
            conf_loss = 0
            epoch += 1
        # 如果迭代的batch数在lr_steps范围内，则调整学习率
        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        # 加载数据集   batch_size=32
        try:
            # target：一行内容是boxes坐标+类别
            images, targets = next(batch_iterator)
        except StopIteration:  # 遇到StopIteration，即迭代完数据集，则再重头开始迭代
            data_loader = data.DataLoader(dataset, args.batch_size,
                                          num_workers=args.num_workers,
                                          shuffle=True, collate_fn=detection_collate,
                                          pin_memory=True)
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)
        # 将训练集转到GPU上
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward前向传播开始
        t0 = time.time()
        out = net(images)
        # backprop反向传播开始
        # 优化器梯度清零
        optimizer.zero_grad()
        # 损失函数得到定位误差和分类误差
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        # 反向传播
        loss.backward()
        # 更新可学习参数
        optimizer.step()
        t1 = time.time()

        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]
        # 每10个batch迭代输出
        if iteration % 10 == 0:
            print('前向+反向总共所需时间timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
        # 可视化
        if args.visdom:
            update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
                            iter_plot, epoch_plot, 'append')
        # 每5000次迭代保存模型
        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_COCO_' +
                       repr(iteration) + '.pth')
    # 保存最后的模型
    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
        调整学习率
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    '''
    初始化新增层的权重
    '''
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    viz = visdom.Visdom()
    '''
    新增可视化图形
    '''
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    '''
    可视化图形里更新数据
    '''
    viz = visdom.Visdom()
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration

    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
