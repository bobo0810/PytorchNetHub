# -*- coding:utf-8 -*-
# power by Mr.Li
from papersReproduced.config import opt
import os
import torch as t
import papersReproduced.models.NetWork as NetWork
from papersReproduced.data.dataprocessing import DataProcessing


from torch.utils.data import DataLoader  #数据加载器
from torch.autograd import Variable
from torchnet import meter  #仪表  用来显示loss等图形
from papersReproduced.utils.visualize import Visualizer  #可视化visdom
from tqdm import tqdm  #显示进度条
from torch.nn import functional as F

def train():
    vis=Visualizer(opt.env)
    netWork=NetWork()
    map_location = lambda storage, loc: storage
    if opt.load_model_path:
        netWork.load_state_dict(t.load(opt.load_model_path,map_location=map_location))
    if opt.use_gpu:
        netWork.cuda(1)
    # step2: 加载数据
    train_data=DataProcessing(opt.data_root,train=True)
    #train=False  test=False   则为验证集
    val_data=DataProcessing(opt.data_root,train=False)
    # 数据集加载器
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, 1, shuffle=False, num_workers=opt.num_workers)
    # step3: criterion and optimizer \
    criterion = t.nn.MSELoss()
    lr=opt.lr
    optimizer = t.optim.Adam(netWork.parameters(), lr=opt.lr, weight_decay =opt.weight_decay)
    # step4: 统计指标meters  仪表 显示损失的图形
    #计算所有书的平均数和标准差，来统计一个epoch中损失的平均值
    loss_meter=meter.AverageValueMeter()
    previous_loss = 1e100
    for epoch in range(opt.max_epoch):
        #清空仪表信息
        loss_meter.reset()
        # 迭代数据集加载器
        for ii, (data_origin, data_grayscale) in enumerate(train_dataloader):
            #训练模型
            input_img=Variable(data_grayscale)
            from visdom import Visdom
            # viz = Visdom()
            # viz.images(
            #     input_img.data.numpy(),
            #     opts=dict(title='input_img', caption='input_img的一个批次图片')
            # )
            #
            output_real_img=Variable(data_origin)
            # viz.images(
            #     output_real_img.data.numpy(),
            #     opts=dict(title='output_real_img', caption='output_real_img的一个批次图片')
            # )


            if opt.use_gpu:
                input_img=input_img.cuda(1)
                output_real_img=output_real_img.cuda(1)
            #优化器梯度清零
            optimizer.zero_grad()
            output_img=netWork(input_img)
            loss=criterion(output_img,output_real_img)
            # 反向传播  自动求梯度         loss进行反向传播
            loss.backward()
            # 更新优化器的可学习参数       optimizer优化器进行更新参数
            optimizer.step()
            # 更新仪表 并可视化
            loss_meter.add(loss.data[0])
            # 每print_freq次可视化loss
            if ii % opt.print_freq == opt.print_freq - 1:
                # plot是自定义的方法
                vis.plot('loss', loss_meter.value()[0])
        # netWork.save()
        # 使用验证集和可视化
        val_output_img= val(netWork, val_dataloader)
        vis.img("val_output_img",val_output_img.cpu().data.numpy())
        # 当前时刻的一些信息
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0],lr=lr))
        # 更新学习率  如果损失开始升高，则降低学习率
        if loss_meter.value()[0] > previous_loss:
            lr=lr*opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]





def val(model,dataloader):
    '''
    计算模型在验证集上的准确率等信息
    '''

    # 将模型调整为验证模式
    model.eval()

    for ii, data in tqdm(enumerate(dataloader)):
        data_origin, data_grayscale = data
        #设置为验证模式
        val_input_data_grayscale=Variable(data_grayscale,volatile=True)
        val_lable_data_origin=Variable(data_grayscale,volatile=True)
        if opt.use_gpu:
            val_input_data_grayscale = val_input_data_grayscale.cuda(1)
            val_lable_data_origin = val_lable_data_origin.cuda(1)
        val_output_img = model(val_input_data_grayscale)

    # 将模型调整为训练模式
    model.train()

    return val_output_img

if __name__ == '__main__':
    train()