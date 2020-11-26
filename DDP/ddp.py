import os
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', default=2, type=int) # 节点数量
    parser.add_argument('--gpus', default=2, type=int) # 每个节点的GPU数量
    parser.add_argument( '--nr', default=0, type=int) # 当前节点在所有节点的序号
    parser.add_argument('--batch', default=128, type=int) # 总batch(有效batch) 均分给全部GPU
    parser.add_argument('--ip',default=None,type=str) # 主节点ip

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes   #总的world_size,即进程总数==总GPU数量（每个进程负责一个GPU）
    os.environ['MASTER_ADDR'] = args.ip        # 主节点（主进程），用于所有进程同步梯度
    os.environ['MASTER_PORT'] = '8886'         # 主进程用于通信的端口，可随意设置

    # 一个节点启动 该节点的所有进程，每个进程运行train(i,args)  i从0到args.gpus-1
    # nprocs：作用于mp.spawn，标明启动的线程数
    # args：传递给train方法的参数
    mp.spawn(train, nprocs=args.gpus, args=(args,))


def train(pid, args):
    '''
    通过mp.spawn启动多进程，train接收参数为：节点内部的子进程号pid + 方法参数
    '''
    # 每个进程负责一个GPU，故 节点内部子进程号 = 节点内部GPU序号
    gpu=pid

    # 计算当前进程在所有进程中的全局排名，每个进程都需要知道进程总数和在进程中的顺序，以便使用哪块GPU
    # rank=0为主进程，用于保存模型和打印信息
    rank = args.nr * args.gpus + gpu

    # 初始化分布环境
    # env：环境变量初始化，需要在环境变量配置4个参数：MASTER_PORT，MASTER_ADDR，WORLD_SIZE，RANK
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=args.world_size,
                            rank=rank)

    torch.manual_seed(0)
    model = ConvNet()

    # 加载权重
    if args.savepath:
        print('loading weights')
        pass

    # DDP分发之前，同步BN（将网络内部的BatchNorm层转换为SyncBatchNorm层）
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    torch.cuda.set_device(gpu) # 当前节点负责的GPU
    model.cuda(gpu)
    batch_size = int(args.batch/args.world_size) # 总的有效batch_size= 均分每块GPU的batch * 总进程数（总GPUs）

    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    # GPU模型包装为 DDP模型
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # 加载数据
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    # 采样器：将数据集分为 world_size 块，不同块送到各进程中
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False, # DDP下该参数无效，由train_sampler负责
                                               num_workers=0, # DDP下为0 否则读取出错
                                               pin_memory=True,
                                               sampler=train_sampler) # 采样器

    for epoch in range(10):
        # 每轮采样器打乱数据集，保证数据划分不同
        train_sampler.set_epoch(epoch)

        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, 10, i + 1, len(train_loader),
                                                                         loss.item()))

        # ===验证===
        # 确保每个进程log名称不同，最后可视化rank=0的log即可
        # acc=eval()


        # 仅主进程 保存模型
        if rank == 0:
            torch.save(model.state_dict(),'ddp.pth')


if __name__ == '__main__':
    main()