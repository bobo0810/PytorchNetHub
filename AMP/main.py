from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import torch
from net import MyNet

def start_train():
    '''
    训练
    '''
    use_amp=True
    # 前向反传N次，再更新参数  目的：增大batch（理论batch= batch_size * N）
    iter_size=8

    myNet = MyNet(use_amp).to("cuda:0")
    myNet = torch.nn.DataParallel(myNet,device_ids=[0,1])
    myNet.train()
    scaler = GradScaler() if use_amp else None

    for epoch in range(1,100):
        for batch_idx, (input, target) in enumerate(dataloader_train):

            # 数据 转到每个并行模型的主卡上
            input = input.to("cuda:0")
            target = target.to("cuda:0")

            # 自动混合精度训练
            if use_amp:
                # 自动广播 将部分Tensor由FP32自动转为FP16
                with autocast():
                    # 提取特征
                    feature=myNet(input)
                    losses = loss_function(target,feature)
                    loss = losses / iter_size
                scaler.scale(loss).backward()
            else:
                feature = myNet(input, target)
                losses = loss_function(target, feature)
                loss = losses / iter_size
                loss.backward()

            # 梯度累积,再更新参数
            if (batch_idx + 1) % iter_size == 0:
                # 梯度更新
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                # 梯度清零
                optimizer.zero_grad()

def start_test():
    '''
    测试 两种情况，哪个性能好选哪个
    情况1：MyNet开启AMP，且应用with autocast()   仅支持英伟达GPU
    情况2：MyNet不开启，图像与模型直接half()
    '''
    use_amp=True
    ##########情况1##########
    # 初始化网络并加载预训练模型
    myNet = MyNet(use_amp).to("cuda:0")
    myNet.eval()
    with torch.no_grad():
        for i, (img,_) in enumerate(test_dataloader):
            img=img.to("cuda:0")
            # 开启半精度预测
            if use_amp:
                with autocast():
                    feature = myNet(input)
            else:
                feature = myNet(input)

    ##########情况2##########
    # 初始化网络并加载预训练模型
    myNet = MyNet()
    myNet.eval()
    with torch.no_grad():
        for i, (img,_) in enumerate(test_dataloader):
            img=img.to("cuda:0")
            # 开启半精度预测
            if use_amp:
                img = img.half()
                myNet = myNet.half()
            feature = myNet(img)


