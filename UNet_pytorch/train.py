import sys
import os
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
from utils.config import opt_train

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,  # 训练集：验证集= 0.95： 0.05
              save_cp=True,
              gpu=False,
              img_scale=0.5):

    dir_img = opt_train.dir_img
    dir_mask = opt_train.dir_mask
    dir_checkpoint = opt_train.dir_checkpoint

    # 得到 图片路径列表  ids为 图片名称（无后缀名）
    ids = get_ids(dir_img)
    # 得到truple元组  （无后缀名的 图片名称，序号）
    # eg:当n为2  图片名称为bobo.jpg 时, 得到（bobo,0） （bobo,1）
    # 当序号为0 时，裁剪宽度，得到左边部分图片  当序号为1 时，裁剪宽度，得到右边部分图片
    ids = split_ids(ids)
    # 打乱数据集后，按照val_percent的比例来 切分 训练集 和 验证集
    iddataset = split_train_val(ids, val_percent)


    print('''
    开始训练:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        训练集大小: {}
        验证集大小: {}
        GPU: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(gpu)))

    #训练集大小
    N_train = len(iddataset['train'])

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    #二进制交叉熵
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))

        # reset the generators
        # 每轮epoch得到 训练集  和 验证集
        train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)




        # 重置epoch损失计数器
        epoch_loss = 0

        for i, b in enumerate(batch(train, batch_size)):
            # 得到 一个batch的 imgs tensor 及 对应真实mask值
            # 当序号为0 时，裁剪宽度，得到左边部分图片[384,384,3]   当序号为1 时，裁剪宽度，得到右边部分图片[384,190,3]
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])

            # 将值转为 torch tensor
            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            # 训练数据转到GPU上
            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            # 得到 网络输出的预测mask [10,1,384,384]
            masks_pred = net(imgs)
            # 经过sigmoid
            masks_probs = F.sigmoid(masks_pred)
            masks_probs_flat = masks_probs.view(-1)

            true_masks_flat = true_masks.view(-1)
            # 计算二进制交叉熵损失
            loss = criterion(masks_probs_flat, true_masks_flat)
            # 统计一个epoch的所有batch的loss之和，用以计算 一个epoch的 loss均值
            epoch_loss += loss.item()

            # 输出 当前epoch的第几个batch  及 当前batch的loss
            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))

            # 优化器梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

        # 一轮epoch结束，该轮epoch的 loss均值
        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        # 每轮epoch之后使用验证集进行评价
        if True:
            # 评价函数：Dice系数   Dice距离用于度量两个集合的相似性
            val_dice = eval_net(net, val, gpu)
            print('Validation Dice Coeff: {}'.format(val_dice))

        # 保存模型
        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))





if __name__ == '__main__':

    # 获取训练参数
    args = opt_train
    # n_channels：输入图像的通道数   n_classes：二分类
    net = UNet(n_channels=3, n_classes=1)

    # 加载预训练模型
    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    # 网络转移到GPU上
    if args.gpu:
        net.cuda()
        cudnn.benchmark = True # 速度更快，但占用内存更多

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        # 当运行出错时，保存最新的模型
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
