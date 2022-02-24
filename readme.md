

# 目的

- 论文复现
- 算法竞赛
- 项目注释 

- Pytorch指北 
- 常用工具

# 工作

### [轻量级图像分类框架Classification](https://github.com/bobo0810/Classification)

> 持续更新

### [Pytorch指北](https://github.com/bobo0810/PytorchGuide)

> 最小实践：自动混合精度AMP、分布式数据并行DDP、TensorRT加速、移动端NCNN/MNN
>
> 模型统计:  计算量|参数量|耗时   

### 视觉任务

> 目标检测：Faster rcnn、YOLO v1、YOLO v3、YOLO v4、SSD、CAM、S4ND
>
> 语义分割：U-Net 
>
> 主干网络：FPN、SKNet、HS-ResNet、AFF-ResNet(并入官方库)、TargetDrop
>
> 激活函数：FunnelAct、DynamicReLU
>
> 知识蒸馏：RepDistiller
>

### 人脸相关

> 人脸检测：Yolov5-Face <u>主要贡献者</u>
>
> 人脸比对
>
> ​		分类器：AMSoftmax|ArcFace|Circle-Loss|DiscFace|NPCFace
>
> ​		训练策略：Semi-Siamese-Training|BroadFace

### 算法竞赛

> 雪浪制造AI挑战赛    排名: 32/2403
>
> “计图”人工智能算法挑战赛-狗细分类  排名: 4/430


# 汇总

## 2022

| 自研                                                         | 备注               | 更新   |
| ------------------------------------------------------------ | ------------------ | ------ |
| [ToolsLib](https://github.com/bobo0810/ToolsLib)             | 常用工具库         | 2022.2 |
| [Classification](https://github.com/bobo0810/Classification) | 轻量级图像分类框架 | 2022.2 |

## 2021

|                         官方库                         |                           项目注释                        |     备注     |  更新  |
| :----------------------------------------------------: | :-------------------------------------------------------: | :----------: | :----: |
| [SST(ECCV 2020)](https://github.com/dituu/Semi-Siamese-Training) | [注释](https://github.com/bobo0810/Semi-Siamese-Training) | 浅层人脸学习 | 2021.2 |
|   [RepDistiller](https://github.com/HobbitLong/RepDistiller)   |     [注释](https://github.com/bobo0810/RepDistiller)      |  知识蒸馏算法合集  | 2021.2 |

|                             自研                             | 备注                                            | 更新    |
| :----------------------------------------------------------: | ----------------------------------------------- | ------- |
| [JittorDogsClass](https://github.com/bobo0810/JittorDogsClass) | “计图”算法挑战赛-狗细分类 4/430                 | 2021.4  |
|   [Yolov5-Face](https://github.com/deepcam-cn/yolov5-face)   | 人脸检测-支持全TensorRT加速，成为**主要贡献者** | 2021.12 |



## 2020

|官方库|项目注释|备注|更新|
|:----:|:---:|:------:|:------:|
|[DynamicReLU(ECCV 2020)](https://github.com/Islanna/DynamicReLU)|[注释](https://github.com/bobo0810/DynamicReLU)|动态激活函数|2020.9|
|[AMSoftmax](https://github.com/cavalleria/cavaface.pytorch)|[注释](https://github.com/bobo0810/FaceVerLoss)|乘法角间隔|2020.9|
|[ArcFace(CVPR 2019)](https://github.com/cavalleria/cavaface.pytorch)|[注释](https://github.com/bobo0810/FaceVerLoss)|加法角间隔|2020.9|
|[CircleLoss(CVPR 2020)](https://github.com/xialuxi/CircleLoss_Face)|[注释](https://github.com/bobo0810/FaceVerLoss)|加权角间隔|2020.9|

|                             自研                             |                   备注                   | 更新    |
| :----------------------------------------------------------: | :--------------------------------------: | ------- |
| [FunnelRelu(ECCV 2020)](https://github.com/bobo0810/FunnelAct_Pytorch) |            新型激活函数-复现             | 2020.7  |
|       [AMP](https://github.com/bobo0810/PytorchGuide)        |            自动混合精度-示例             | 2020.10 |
| [BroadFace(ECCV 2020)](https://github.com/bobo0810/BroadFace) |        人脸对比队列更新策略-复现         | 2020.10 |
|     [TargetDrop](https://github.com/bobo0810/TargetDrop)     |          注意力机制Dropout-复现          | 2020.10 |
|      [HS-ResNet](https://github.com/bobo0810/HS-ResNet)      |            ResNet改进版-复现             | 2020.11 |
| [AFF-ResNet(WACV 2021)](https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/README_CN.md) | 特征融合的统一方式- 复现，**并入官方库** | 2020.11 |
|       [DDP](https://github.com/bobo0810/PytorchGuide)        |           分布式数据并行-示例            | 2020.11 |
| [DiscFace(ACCV 2020)](https://github.com/bobo0810/FaceVerLoss) |            最小差异学习-复现             | 2020.12 |
|      [NPCFace](https://github.com/bobo0810/FaceVerLoss)      |            正负联合监督-复现             | 2020.12 |



## 2017-2019
|官方库|项目注释|备注|
|:----:|:---:|:------:|
|[Fatser Rcnn(NIPS 2015)](https://zhuanlan.zhihu.com/p/32404424)|[注释](https://github.com/bobo0810/PytorchNetHub/tree/master/FasterRcnn_pytorch)|目标检测|
|[YOLO v1(CVPR 2016)](https://github.com/xiongzihua/pytorch-YOLO-v1)|[注释](https://github.com/bobo0810/PytorchNetHub/tree/master/Yolov1_pytorch)|目标检测|
|[YOLO v3(ECCV 2016)](https://github.com/eriklindernoren/PyTorch-YOLOv3)|[注释](https://github.com/bobo0810/PyTorch-YOLOv3-master) [重构](https://github.com/bobo0810/PytorchNetHub/tree/master/Yolov3_pytorch)|目标检测|
|[YOLO v4](https://github.com/Tianxiaomo/pytorch-YOLOv4)|[注释](https://github.com/bobo0810/YOLOv4_Pytorch)|目标检测|
|[SSD](https://github.com/amdegroot/ssd.pytorch)|[注释](https://github.com/bobo0810/pytorchSSD)   [重构](https://github.com/bobo0810/PytorchNetHub/tree/master/SSD_pytorch)|目标检测|
|[CAM(CVPR 2016)](https://github.com/jacobgil/keras-cam)|[注释](https://github.com/bobo0810/PytorchNetHub/tree/master/CAM_pytorch)|特征可视化|
|[U-Net(MICCAI 2015)](https://github.com/milesial/Pytorch-UNet)|[注释](https://github.com/bobo0810/PytorchNetHub/tree/master/UNet_pytorch) |医学影像语义分割|
|[FPN(CVPR 2017)](https://github.com/kuangliu/pytorch-fpn)|[注释](https://github.com/bobo0810/PytorchNetHub/tree/master/FPN_pytorch)|特征金字塔|

|                             自研                             | 备注                                           |
| :----------------------------------------------------------: | ---------------------------------------------- |
| [XueLangTianchi](https://github.com/bobo0810/XueLangTianchi) | 雪浪制造AI挑战赛—视觉计算辅助良品检测 -32/2403 |
|       [S4ND](https://github.com/bobo0810/S4ND_Pytorch)       | 单次单尺度肺结节检测（MICCAI 2018）复现        |
|    [3D SKconv](https://github.com/bobo0810/SKNet_Pytorch)    | 注意力机制SE模块、SK模块的3D实现               |


> 注：猫狗分类、风格迁移、生成对抗等更多内容请访问[pytorch-book](https://github.com/chenyuntc/pytorch-book)

