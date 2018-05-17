# -*- coding:utf-8 -*-
# power by Mr.Li
class DefaultConfig():
    env = 'default'  # visdom 环境
    model = 'NetWork'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    data_root='/home/bobo/data/PapersReproduced/'


    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 32  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    max_epoch = 10
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数
opt=DefaultConfig()