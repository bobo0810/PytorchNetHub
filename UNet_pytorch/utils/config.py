# -*- coding:utf-8 -*-
# power by Mr.Li
# 设置默认参数
import os.path
class DefaultConfig_train():
    epochs=5  #number of epochs
    batchsize= 10  #batch size
    lr=0.1  #learning rate
    gpu=True  #use cudas
    load=False  #load file model
    scale=0.3    #downscaling factor of the images  图像训练时缩小倍数       该值对内存影响较大（仓库默认0.5）

    # 数据集
    dir_img = '/home/bobo/data/CarvanaImageMaskingChallenge_UNet/train/'
    dir_mask = '/home/bobo/data/CarvanaImageMaskingChallenge_UNet/train_masks/'
    dir_checkpoint = './checkpoints/'  # 模型保存位置

    visdom=True   # 是否可视化

    env = 'U-Net'  # visdom 环境的名字
    visdom = True  # 是否可视化
    datesets_name='Carvana Image Masking Challenge'  # 数据集名称







class DefaultConfig_predict():
    input='./intput.jpg'    #filenames of input images
    output='./output.jpg'   #filenames of ouput images
    model= './MODEL.pth'    # Specify the file in which is stored the model
    cpu=False   #Do not use the cuda version of the net
    scale=0.5    #Scale factor for the input images
    mask_threshold=0.5   #Minimum probability value to consider a mask pixel white
    no_crf=False    #Do not use dense CRF postprocessing
    no_save=False   #Do not save the output masks
    viz=False    #Visualize the images as they are processed
#初始化该类的一个对象
opt_train=DefaultConfig_train()

opt_predict=DefaultConfig_predict()

