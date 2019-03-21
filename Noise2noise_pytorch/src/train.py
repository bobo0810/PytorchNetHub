#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from datasets import load_dataset
from noise2noise import Noise2Noise
from argparse import ArgumentParser
import os

def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data parameters
    parser.add_argument('-t', '--train-dir', help='training set path', default='/home/bobo/data/noise2noise/train')  # 训练集地址
    parser.add_argument('-v', '--valid-dir', help='test set path', default='/home/bobo/data/noise2noise/valid') #验证集地址
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='./../ckpts') #模型保存地址
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', action='store_true') #仅保存最新模型
    parser.add_argument('--report-interval', help='batch report interval', default=250, type=int) #每几次进行汇报
    parser.add_argument('-ts', '--train-size', help='size of train dataset', default=1000,type=int) #训练集大小
    parser.add_argument('-vs', '--valid-size', help='size of valid dataset', default=200,type=int) # 验证集大小

    # Training hyperparameters 训练超参数
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.001, type=float) # 初始学习率
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list) # 优化器
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=4, type=int) # batch-size
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=100, type=int) # 训练轮数
    parser.add_argument('-l', '--loss', help='loss function', choices=['l1', 'l2', 'hdr'], default='l1', type=str) #损失函数  选L1，倾向于学习中位数
    parser.add_argument('--cuda', help='use cuda', action='store_true', default=True) # 使用GPU
    parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_true') #每轮画损失图

    # Corruption parameters
    parser.add_argument('-n', '--noise-type', help='noise type',
        choices=['gaussian', 'poisson', 'text', 'mc'], default='text', type=str)  # 选择噪声类型，选水印
    parser.add_argument('-p', '--noise-param', help='noise parameter (e.g. std for gaussian)', default=0.5, type=float)  # text噪声选0.5
    parser.add_argument('-s', '--seed', help='fix random seed', type=int) # 固定随机种子
    parser.add_argument('-c', '--crop-size', help='random crop size', default=128, type=int) # 随机裁剪的尺寸
    parser.add_argument('--clean-targets', help='use clean targets for training', action='store_true') # 使用干净目标进行训练

    return parser.parse_args()


if __name__ == '__main__':
    """Trains Noise2Noise."""

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # Parse training parameters
    params = parse_args()

    # Train/valid datasets  加载数据集
    train_loader = load_dataset(params.train_dir, params.train_size, params, shuffled=True)
    valid_loader = load_dataset(params.valid_dir, params.valid_size, params, shuffled=False)

    # Initialize model and train  初始化模型并训练
    n2n = Noise2Noise(params, trainable=True)
    n2n.train(train_loader, valid_loader)
