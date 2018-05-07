#!/usr/bin/python
# -*- coding:utf-8 -*-
# power by Mr.Li
from config import opt
import os
import torch as t
import models
from data.dataset import DogCat   #加载转换后的数据集
from torch.utils.data import DataLoader  #数据加载器
from torch.autograd import Variable
from torchnet import meter  #仪表  用来显示loss等图形
from utils.visualize import Visualizer  #可视化visdom
from tqdm import tqdm  #显示进度条
def test(**kwargs):
    '''
    猫狗大战 测试集 预测完成后写入cvs中每张图片为狗的概率
    '''
    opt.parse(kwargs)
    # 通过config中模型名称来加载模型
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()
    # data
    train_data = DogCat(opt.test_data_root, test=True)
    test_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    results = []
    # 测试集命名只有数字  data/test1/8973.jpg    path为8973
    for ii, (data, path) in tqdm(enumerate(test_dataloader)):
        input = t.autograd.Variable(data, volatile=True)
        if opt.use_gpu:
            input = input.cuda()
        score = model(input)
        #概率  通过softmax可得概率   [:, 0]行全要，列只要第一列
        probability = t.nn.functional.softmax(score)[:, 0].data.tolist()
        #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。 如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同（两个对象 path,probability）
        #两个对象 path,probability  逐元素拿出来打包成一个元组，返回由这些元组组成的列表
        batch_results=[(path_,probability_) for path_,probability_  in zip(path,probability)]
        results=results+batch_results
    #result_file 写入的文件地址
    write_csv(results,opt.result_file)
def write_csv(results,file_name):
    import csv
    #调整为写入模式
    with open(file_name,'w') as f:
        writer=csv.writer(f)
        # 写入标题
        writer.writerow(['id','label'])
        #写入元组数据
        writer.writerows(results)

def train(**kwargs):
    opt.parse(kwargs)
    vis= Visualizer(opt.env)
    # step1:  通过config中模型名称来加载模型
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

   # step2: 加载数据
    #封装数据集
    train_data=DogCat(opt.train_data_root,train=True)
    #train=False  test=False   则为验证集
    val_data=DogCat(opt.train_data_root,train=False)
    #数据集加载器
    train_dataloader=DataLoader(train_data,opt.batch_size,shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,opt.batch_size,shuffle=False,num_workers=opt.num_workers)
    # step3: criterion and optimizer   分类都用交叉熵   优化器用adam
    criterion = t.nn.CrossEntropyLoss()
    optimizer=t.optim.Adam(model.parameters(),lr=opt.lr,weihtt_decay=opt.weight_decay)
    # step4: 统计指标meters  仪表 显示损失的图形
    #计算所有书的平均数和标准差，来统计一个epoch中损失的平均值
    loss_meter=meter.AverageValueMeter()
    #混淆矩阵
    #2分类  统计分类问题中的分类情况
    confusion_matrix=meter.ConfusionMeter(2)
    previous_loss=1e100

    #开始训练
    for epoch in range(opt.max_epoch):
        #清空仪表信息和混淆矩阵信息
        loss_meter.reset()
        confusion_matrix.reset()
        #迭代数据集加载器
        for ii,(data,label) in tqdm(enumerate(train_dataloader)):
            #训练模型
            input=Variable(data)
            target=Variable(label)
            if opt.use_gpu:
                input=input.cuda()
                target = target.cuda()
            #优化器梯度清零
            optimizer.zero_grad()
            score=model(input)
            loss=criterion(score,target)
            #反向传播  自动求梯度         loss进行反向传播
            loss.backward()
            #更新优化器的可学习参数       optimizer优化器进行更新参数
            optimizer.step()

            #更新仪表 并可视化
            loss_meter.add(loss.data[0])
            confusion_matrix.add(score.data,target.data)
            # 每print_freq次可视化loss
            if ii % opt.print_freq == opt.print_freq - 1:
                #plot是自定义的方法
                vis.plot('loss',loss_meter.value()[0])
        #保存模型
        model.save()
        #使用验证集验证和可视化   val_cm：混淆矩阵的值，是一个矩阵   val_accuracy：正确率，是一个数
        val_cm,val_accuracy=val(model,val_dataloader)
        vis.plot('val_accuracy',val_accuracy)
        #当前时刻的一些信息
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
                    epoch = epoch,loss = loss_meter.value()[0],val_cm = str(val_cm.value()),train_cm=str(confusion_matrix.value()),lr=lr))
        #更新学习率  如果损失开始升高，则降低学习率
        if loss_meter.value()[0]>previous_loss:
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] =  lr * opt.lr_decay
        previous_loss=loss_meter.value()[0]



def val(model,dataloader):
    '''
    计算模型在验证集上的准确率等信息
    '''
    model.eval()
    # 将模型调整为验证模式
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in tqdm(enumerate(dataloader)):
        input, label = data
        #设置为验证模式
        val_input=Variable(input,volatile=True)
        val_lable=Variable(label.type(t.LongTensor),volatile=True)
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input)
        confusion_matrix.add(score.data.squeeze(),label.type(t.LongTensor))
    # 将模型调整为训练模式
    model.train()
    #cm_value   混淆矩阵的值
    cm_value=confusion_matrix.value()
    # 预测正确的数量除以总数量 再*100  得到正确率
    accuracy=100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy




