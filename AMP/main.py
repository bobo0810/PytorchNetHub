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
    myNet = torch.nn.DataParallel(myNet,device_ids=[0,1]) # 数据并行
    myNet.train()
    # 训练开始前初始化 梯度缩放器
    scaler = GradScaler() if use_amp else None

    # 加载预训练权重
    if resume_train:
        scaler.load_state_dict(checkpoint['scaler']) # amp自动混合精度用到
        optimizer.load_state_dict(checkpoint['optimizer'])
        myNet.load_state_dict(checkpoint["model"])


    for epoch in range(1,100):
        for batch_idx, (input, target) in enumerate(dataloader_train):

            # 数据 转到每个并行模型的主卡上
            input = input.to("cuda:0")
            target = target.to("cuda:0")

            # 自动混合精度训练
            if use_amp:
                # 自动广播 将支持半精度操作自动转为FP16
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
        # scaler 具有状态。恢复训练时需要加载
        state = {'net': myNet.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict()}
        torch.save(state, "filename.pth")

def start_test():
    '''
    测试
    '''
    # 初始化网络并加载预训练模型
    myNet = MyNet().to("cuda:0")
    myNet.eval()
    with torch.no_grad():
        input = input.to("cuda:0")

        # 若想推理加速，在精度接受范围内img\model手动half()为FP16，然后只能GPU推理
        # input=input.half()
        # myNet=myNet.half()
        feature = myNet(input)





