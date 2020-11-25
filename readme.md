# 目的

- 项目注释 
  > 即开即用推荐原仓库,阅读源码推荐注释
- 论文复现

# 接下来工作

### 目标检测
- [x] Faster rcnn
- [x] YOLO v1
- [x] YOLO v3
- [x] YOLO v4
- [x] SSD
- [x] CAM
- [x] S4ND


### 语义分割
- [x] U-Net

### 人脸识别
- [x] AMSoftmax
- [x] ArcFace
- [x] Circle-Loss
- [x] Semi-Siamese-Training
- [x] BroadFace
- [x] TargetDrop

### 主干网络
- [x] HS-ResNet
- [x] AFF-ResNet [并入官方库](https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/README_CN.md)


### 激活函数
- [x] FunnelAct
- [x] DynamicReLU

### 其他
- [x] SKNet
- [x] FPN
- [x] AMP
- [x] DDP



# 汇总



## 2020
|模型|原仓库|加注释|实现|来源|备注|更新|
|:---:|:----:|:---:|:------:|:------:|:------:|:------:|
|[FunnelRelu](https://arxiv.org/pdf/2007.11824.pdf)|[MegEngine版](https://github.com/megvii-model/FunnelAct) ||[复现](https://github.com/bobo0810/FunnelAct_Pytorch)|ECCV 2020|较PRelu等更有效|2020.7|
|[DynamicReLU](https://arxiv.org/abs/2003.10027)|[原地址](https://github.com/Islanna/DynamicReLU)|[注释](https://github.com/bobo0810/DynamicReLU)||ECCV 2020|动态激活函数|2020.9|
|[AMSoftmax](https://arxiv.org/pdf/1801.05599.pdf)|[原地址](https://github.com/cavalleria/cavaface.pytorch)|[注释](https://github.com/bobo0810/FaceVerLoss)|||乘法角间隔|2020.9|
|[ArcFace](https://arxiv.org/abs/1801.07698)|[原地址](https://github.com/cavalleria/cavaface.pytorch)|[注释](https://github.com/bobo0810/FaceVerLoss)||CVPR 2019|加法角间隔|2020.9|
|[CircleLoss](https://arxiv.org/abs/2002.10857)|[原地址](https://github.com/xialuxi/CircleLoss_Face)|[注释](https://github.com/bobo0810/FaceVerLoss)||CVPR 2020 Oral|加权角间隔|2020.9|
|[SST](https://arxiv.org/abs/2007.08398)|[原地址](https://github.com/dituu/Semi-Siamese-Training)|[注释](https://github.com/bobo0810/Semi-Siamese-Training)||ECCV 2020|浅层人脸学习|2020.10|
|AMP|||[实现](https://github.com/bobo0810/PytorchNetHub/tree/master/AMP)||自动混合精度|2020.10|
|[BroadFace](https://arxiv.org/abs/2008.06674)|||[复现](https://github.com/bobo0810/BroadFace)|ECCV 2020|队列更新|2020.10|
|[TargetDrop](https://arxiv.org/abs/2010.10716)|||[复现](https://github.com/bobo0810/TargetDrop)||注意力Drop|2020.10|
|[HS-ResNet](https://arxiv.org/abs/2010.07621)|||[复现](https://github.com/bobo0810/HS-ResNet)||改进ResNet|2020.11|
|[AFF-ResNet](https://arxiv.org/abs/2009.14082)|[MXNet版](https://github.com/YimianDai/open-aff)||[复现](https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/README_CN.md)|WACV 2021|统一特征融合|2020.11|
|DDP|||[实现](https://github.com/bobo0810/PytorchNetHub/tree/master/DDP)||分布式数据并行|2020.11|



## 2017-2019
|模型|原仓库|加注释|实现|来源|
|:---:|:----:|:---:|:------:|:------:|
|[Fatser Rcnn](https://arxiv.org/abs/1506.01497) |[原地址](https://zhuanlan.zhihu.com/p/32404424)|[注释](https://github.com/bobo0810/PytorchNetHub/tree/master/FasterRcnn_pytorch)||NIPS 2015||
|[YOLO v1](https://arxiv.org/abs/1506.02640) |[原地址](https://github.com/xiongzihua/pytorch-YOLO-v1)|[注释](https://github.com/bobo0810/PytorchNetHub/tree/master/Yolov1_pytorch)||CVPR 2016|
|[YOLO v3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) |[原地址](https://github.com/eriklindernoren/PyTorch-YOLOv3)|[注释](https://github.com/bobo0810/PyTorch-YOLOv3-master)|[重构](https://github.com/bobo0810/PytorchNetHub/tree/master/Yolov3_pytorch)||
|[YOLO v4](https://arxiv.org/pdf/2004.10934.pdf) |[原地址](https://github.com/Tianxiaomo/pytorch-YOLOv4)|[注释](https://github.com/bobo0810/YOLOv4_Pytorch)|||
|[SSD](https://arxiv.org/abs/1512.02325)|[原地址](https://github.com/amdegroot/ssd.pytorch)|[注释](https://github.com/bobo0810/pytorchSSD)|[重构](https://github.com/bobo0810/PytorchNetHub/tree/master/SSD_pytorch)|ECCV 2016|
|[CAM](https://arxiv.org/pdf/1512.04150.pdf) |[原地址](https://github.com/jacobgil/keras-cam)|[注释](https://github.com/bobo0810/PytorchNetHub/tree/master/CAM_pytorch)||CVPR 2016|
|[S4ND](https://arxiv.org/pdf/1805.02279.pdf?fbclid=IwAR0B3dI8tjvWz-Mk9Xpyymfnk-SNs6k8tw2B8HU3dTTP-vFinQURHGZSCQs) |||[复现](https://github.com/bobo0810/S4ND_Pytorch)|MICCAI 2018|
|[U-Net](https://arxiv.org/abs/1505.04597)|[原地址](https://github.com/milesial/Pytorch-UNet)|[注释](https://github.com/bobo0810/PytorchNetHub/tree/master/UNet_pytorch) ||MICCAI 2015|
|[SKNet](https://arxiv.org/pdf/1903.06586.pdf)|[原地址](https://github.com/implus/SKNet)||[实现](https://github.com/bobo0810/SKNet_Pytorch)|CVPR 2019|

|[FPN](https://arxiv.org/abs/1612.03144)|[原地址](https://github.com/kuangliu/pytorch-fpn)|[注释](https://github.com/bobo0810/PytorchNetHub/tree/master/FPN_pytorch)||CVPR 2017|





注：猫狗大战、风格迁移、GAN生成对抗网络等更多内容请访问[传送门](https://github.com/chenyuntc/pytorch-book)

# 阅读源码三步骤

- 数据预处理
- 网络模型搭建
- 损失函数定义

# 项目结构


- 总结构

  ![](https://github.com/bobo0810/imageRepo/blob/master/img/99053959.jpg)
  
  
- 项目结构

  1、定义网络
  
  ![](https://github.com/bobo0810/imageRepo/blob/master/img/16409622.jpg) 
  
   2、封装数据集
   
  ![](https://github.com/bobo0810/imageRepo/blob/master/img/38894621.jpg)
  
   3、工具类
   
  ![](https://github.com/bobo0810/imageRepo/blob/master/img/98583532.jpg)
  
   4、主函数
   
  ![](https://github.com/bobo0810/imageRepo/blob/master/img/32257225.jpg)

