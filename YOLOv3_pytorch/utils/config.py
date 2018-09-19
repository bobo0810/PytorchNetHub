# -*- coding:utf-8 -*-
# power by Mr.Li
# 设置默认参数
import os.path
class DefaultConfig_train():
    epochs=30    # 训练轮数
    image_folder='data/samples'   #数据集地址
    batch_size=16    #batch大小
    model_config_path='config/yolov3.cfg'   # 模型网络结构
    data_config_path='config/coco.data'    # 配置数据集的使用情况
    class_path='data/coco.names'            #coco数据集类别标签
    conf_thres=0.8                          # 物体置信度阈值
    nms_thres=  0.4            # iou for nms的阈值
    n_cpu=0                 # 批生成期间要使用的cpu线程数
    img_size=416    # 输入图像尺寸的大小
    use_cuda=True     # 是否使用GPU
    visdom=True  # 是否使用visdom来可视化loss
    print_freq = 8  # 训练时，每N个batch显示
    lr_decay = 0.1  # 1e-3 -> 1e-4

    checkpoint_interval=1   # 每隔几个模型保存一次
    checkpoint_dir='./checkpoints'   # 保存生成模型的路径

    load_model_path=None   # 加载预训练的模型的路径，为None代表不加载
    # load_model_path=checkpoint_dir+'/latestbobo.pt'  # 预训练权重

class DefaultConfig_test():
    epochs=200   #number of epochs
    batch_size=16   #size of each image batch
    model_config_path='config/yolov3.cfg'  #'path to model config file'
    data_config_path='config/coco.data'   #'path to data config file'

    checkpoint_dir = './checkpoints'  # 保存生成模型的路径
    # load_model_path=None   # 加载预训练的模型的路径，为None代表不加载
    load_model_path=checkpoint_dir+'/8yolov3.pt'  # 预训练权重

    class_path='data/coco.names'   #'path to class label file'
    iou_thres=0.5  #'iou threshold required to qualify as detected'
    conf_thres=0.5 #'object confidence threshold'
    nms_thres=0.45  #'iou thresshold for non-maximum suppression'
    n_cpu=0   #'number of cpu threads to use during batch generation'
    img_size=416  #size of each image dimension
    use_cuda=True  #'whether to use cuda if available'


class DefaultConfig_detect():
    image_folder= 'data/samples'  #path to dataset
    config_path='config/yolov3.cfg'  #path to model config file


    checkpoint_dir='./checkpoints'   # 保存生成模型的路径
    # load_model_path=None   # 加载预训练的模型的路径，为None代表不加载
    load_model_path = checkpoint_dir + '/yolov3.weights'  # 预训练权重


    class_path='data/coco.names'    #path to class label file
    conf_thres=0.8    #object confidence threshold
    nms_thres=0.4    #iou thresshold for non-maximum suppression
    batch_size=1   #size of the batches
    n_cpu=8   #number of cpu threads to use during batch generation
    img_size=416   #size of each image dimension
    use_cuda=True   #whether to use cuda if available



#初始化该类的一个对象
opt_train=DefaultConfig_train()
opt_test=DefaultConfig_test()
opt_detect=DefaultConfig_detect()
