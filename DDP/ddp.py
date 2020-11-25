import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)') # 节点数量
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node') # 每个节点的GPU数量
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes') # 当前节点在所有节点的序号
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--batch', default=128, type=int) # 总batch(有效batch) 均分给全部GPU
    parser.add_argument('--savepath', default='ddp.pth', type=str)

    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes    #总的world_size,即进程总数==总GPU数量（每个进程一个GPU）
    os.environ['MASTER_ADDR'] = '193.168.1.156' # 主进程，用于所有进程同步
    os.environ['MASTER_PORT'] = '8886' # 进程0的端口
    # 一个节点启动 该节点所有进程，每个进程运行train(i,args)  i从0到 args.gpus-1
    mp.spawn(train, nprocs=args.gpus, args=(args,))


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


def train(pid, args):
    gpu=pid # 单个节点内部的子进程号
    # 初始化分布环境
    # 当args.gpus=2时，gpu取值为0、1
    # 所有进程中 进程的全局排名，每个进程都需要知道进程总数和在进程中的顺序，以便使用哪块GPU
    # rank=0 主进程（保存模型or打印信息）
    rank = args.nr * args.gpus + gpu

    # env：环境变量初始化，需要在环境变量配置4个参数：
        # MASTER_PORT，MASTER_ADDR，WORLD_SIZE，RANK
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=args.world_size,
                            rank=rank)


    # 随机种子必须在运行函数中设置
    torch.manual_seed(0)
    model = ConvNet()

    # 加载权重
    if args.savepath:
        print('loading weights')
    # DDP分发之前，同步BN（将 BatchNorm 层转换为 SyncBatchNorm）
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    torch.cuda.set_device(gpu) # 当前节点的第几块GPU（一个进程一个GPU）
    model.cuda(gpu)
    batch_size = int(args.batch/args.world_size) #均分为每块GPU的batch
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    # Wrap the model  模型先加载到GPU上才能进行分发
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False, # 分布式下该参数无效
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler) # 采样器

    start = datetime.now()
    total_step = len(train_loader)

    # # 打印参数，验证 每个模型初始权重是否相同
    # state_dict = model.state_dict()
    # params = state_dict.items()
    # print(params)

    for epoch in range(args.epochs):
        # 保证每轮采样器的数据划分不同
        train_sampler.set_epoch(epoch)

        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                         loss.item()))

        # ===验证===
        # acc=eval()

        # 每个节点的第一个进程 均保存一份
        # if rank % args.gpus == 0:
        # 仅主进程 保存一份
        if rank == 0:
            # 验证信息写入tensorboard
            # 保存权重
            torch.save(model.state_dict(),'ddp.pth')

    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))


if __name__ == '__main__':

    # 只有 所有节点执行，才开始运行
    #
    # python bobo.py -n 节点数 -g 每个节点的GPU数量 -nr 当前节点序号
    # 主卡156：python bobo.py -n 2 -g 2 -nr 0
    # 其余机器：python bobo.py -n 2 -g 2 -nr 1
    main()