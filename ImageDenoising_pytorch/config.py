# -*- coding:utf-8 -*-
# power by Mr.Li
# 设置默认参数
class DefaultConfig():
    env = 'default'  # visdom 环境的名字
    model = 'NetWork'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    data_root='/home/bobo/data/PapersReproduced/'   #该文件夹下存着两个文件夹  原图  跟  灰度加噪图
    img_lena='/home/bobo/data/PapersReproduced/lena.jpg'  #测试图片，有原图 跟 灰度加噪图


    # load_model_path = "/home/bobo/PycharmProjects/torchProjectss/papersReproduced/checkpoints/NetWork_0517_21:54:05.pth"  # 加载预训练的模型的路径，为None代表不加载
    load_model_path =None

    batch_size = 32  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data  加载数据时的线程
    print_freq = 20  # print info every N batch  

    max_epoch = 100  
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数、
 #初始化该类的一个对象  
opt=DefaultConfig()