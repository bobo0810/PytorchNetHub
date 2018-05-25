# -*- coding:utf-8 -*-
# power by Mr.Li
# 设置默认参数
class DefaultConfig():
    env = 'YOLOv1'  # visdom 环境的名字
    # model = 'NetWork'  # 使用的模型，名字必须与models/__init__.py中的名字一致
    file_root = '/home/zhuhui/data/VOCdevkit/VOC2012/JPEGImages/'  #VOC2012的训练集
    test_root = '/home/zhuhui/data/VOCdevkit/VOC2007/JPEGImages/'   #VOC2007的测试集
    train_Annotations = '/home/zhuhui/data/VOCdevkit/VOC2012/Annotations/'
    voc_2007test='/home/bobo/PycharmProjects/torchProjectss/YOLOv1ByBobo/data/voc2007test.txt'
    voc_2012train='/home/bobo/PycharmProjects/torchProjectss/YOLOv1ByBobo/data/voc2012train.txt'

    test_img_dir='/home/bobo/PycharmProjects/torchProjectss/YOLOv1ByBobo/testImgs/a.jpg'
    result_img_dir='/home/bobo/PycharmProjects/torchProjectss/YOLOv1ByBobo/testImgs/result_a.jpg'



    batch_size = 32  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data  加载数据时的线程
    print_freq = 20  # print info every N batch

    # load_model_path =None  # 加载预训练的模型的路径，为None代表不加载
    best_test_loss_model_path= '/home/bobo/PycharmProjects/torchProjectss/YOLOv1ByBobo/checkpoint/yolo_val_best.pth'
    current_epoch_model_path='/home/bobo/PycharmProjects/torchProjectss/YOLOv1ByBobo/checkpoint/yolo_bobo.pth'
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载
    num_epochs = 120   #训练的epoch次数
    learning_rate = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    momentum=0.95
    weight_decay =5e-4  # 损失函数
    # VOC的类别
    VOC_CLASSES = (  # always index 0
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')


 #初始化该类的一个对象
opt=DefaultConfig()