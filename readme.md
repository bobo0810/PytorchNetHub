![linus](https://user-images.githubusercontent.com/26057879/204754749-ba2705ba-0844-4079-8f5e-2a90d5fb634b.jpg)

# 目的

- 论文复现
- 算法竞赛
- 源码注释
  
   > paper得来终觉浅，绝知此事要coding。



# 工作

### [轻量级图像识别框架Classification](https://github.com/bobo0810/Classification)![Github stars](https://img.shields.io/github/stars/bobo0810/Classification.svg)

> 支持任务： 1. 图像分类   2. 度量学习/特征对比
>
> 轻量级、模块化、高扩展、分布式、自动剪枝

### [工具库bobotools](https://github.com/bobo0810/bobotools)![Github stars](https://img.shields.io/github/stars/bobo0810/bobotools.svg)
> Pytorch工具torch_tools、图像工具img_tools、文本工具txt_tools、列表工具list_tools 

### [Pytorch最小实践](https://github.com/bobo0810/PytorchExample)![Github stars](https://img.shields.io/github/stars/bobo0810/PytorchExample.svg)

> 自动混合精度AMP
> 
> 分布式数据并行DDP
> 
> NCNN/MNN部署
> 
> TensorRT部署


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
> "计图"人工智能算法挑战赛-狗细分类  排名: 4/430
>
> ACCV2022国际细粒度图像分析挑战赛-网络监督的细粒度识别  排名: 8/133



# 时间线



## 2024

| 自研   | 备注                          | 更新   |
| ------ | ----------------------------- | ------ |
| [多模态大模型专栏](https://www.zhihu.com/column/c_1801288994694258689) | 多模态VL系列之高质量数据（一）<br>多模态VL系列之模型架构（二）<br> 多模态VL系列之训练策略（三）<br> | 2024.4 |
| [24卡+公开数据+4B 可以有多强？](https://zhuanlan.zhihu.com/p/714269886) | 勇闯OpenCompass榜单 | 2024.8 |




## 2023

| 自研                                                         | 备注                                                    | 更新   |
| ------------------------------------------------------------ | ------------------------------------------------------- | ------ |
| [LearnColossalAI](https://github.com/bobo0810/LearnColossalAI) | ColossalAI教程 & 示例注释 & 学习笔记 （大模型高效训练） | 2023.3 |
| [OpenMM](https://github.com/bobo0810/OpenMM)                 | OpenMM系列最佳食用手册                                  | 2023.3 |
| [LearnDeepSpeed](https://github.com/bobo0810/LearnDeepSpeed)![Github stars](https://img.shields.io/github/stars/bobo0810/LearnDeepSpeed.svg) | DeepSpeed教程 & 示例注释 & 学习笔记 （大模型高效训练）  | 2023.8 |
| [MiniGPT-4-DeepSpeed](https://github.com/bobo0810/MiniGPT-4-DeepSpeed) | MiniGPT-4基于DeepSpeed加速➕ 扩充模型规模 ➕ 实验分析     | 2023.9 |



## 2022

|                          官方库                          |                         |  更新  |
| :------------------------------------------------------: | :-------------: | :----: |
| [RepVGG(CVPR 2021)](https://github.com/DingXiaoH/RepVGG) | VGG再次伟大  [解读+代码](./RepVGG/readme.md) | 2022.7 |
| [G-Ghost(IJCV 2022)](https://github.com/huawei-noah/Efficient-AI-Backbones) | 探索各级别的特征冗余   [解读+代码](./GhostNet/readme.md) | 2022.7 |
| [BBN(CVPR2020)](https://github.com/megvii-research/BBN) | 双边分支网络,以解决长尾分布问题   [解读+代码](./BBN/readme.md) | 2022.11 |
| Vision Transformer(ICLR 2021) | Transformer首次应用到视觉领域    [解读+代码](https://github.com/bobo0810/LearnTransformer) | 2022.12 |
| [MAE(CVPR2021)](https://github.com/facebookresearch/mae) | 自监督学习   [解读+代码](https://github.com/bobo0810/LearnTransformer) | 2022.12 |

| 自研                                                         | 备注                                 | 更新    |
| ------------------------------------------------------------ | ------------------------------------ | ------- |
| [bobotools](https://github.com/bobo0810/botools)![Github stars](https://img.shields.io/github/stars/bobo0810/botools.svg) | 工具库                               | 2022.2  |
| [Classification](https://github.com/bobo0810/Classification)![Github stars](https://img.shields.io/github/stars/bobo0810/Classification.svg) | 图像识别框架                         | 2022.2  |
| [Pytorch最小实践](https://github.com/bobo0810/PytorchExample)![Github stars](https://img.shields.io/github/stars/bobo0810/PytorchExample.svg) | Pytorch最小实践                      | 2022.6  |
| [BossVision](https://www.yuque.com/bobo0810/boss_vision/guly2g) | 简单、模块化、高扩展的分布式训练框架 | 2022.7  |
| [CUDA-Python](./CUDA_Python/readme.md)                       | Nvidia CUDA加速计算基础课程          | 2022.9  |
| [DataHub](./DataHub/readme.md)                               | 公开数据集汇总                       | 2022.11 |

## 2021

|                              官方库                              |                         项目注释                          |       备注       |  更新  |
| :--------------------------------------------------------------: | :-------------------------------------------------------: | :--------------: | :----: |
| [SST(ECCV 2020)](https://github.com/dituu/Semi-Siamese-Training) | [注释](https://github.com/bobo0810/Semi-Siamese-Training) |   浅层人脸学习   | 2021.2 |
|    [RepDistiller](https://github.com/HobbitLong/RepDistiller)    |     [注释](https://github.com/bobo0810/RepDistiller)      | 知识蒸馏算法合集 | 2021.2 |

|                             自研                             | 备注                                            | 更新    |
| :----------------------------------------------------------: | ----------------------------------------------- | ------- |
| [JittorDogsClass](https://github.com/bobo0810/JittorDogsClass)![Github stars](https://img.shields.io/github/stars/bobo0810/JittorDogsClass.svg) | “计图”算法挑战赛-狗细分类 4/430                 | 2021.4  |
| [Yolov5-Face](https://github.com/deepcam-cn/yolov5-face)![Github stars](https://img.shields.io/github/stars/deepcam-cn/yolov5-face.svg) | 人脸检测-支持纯TensorRT加速，成为**主要贡献者** | 2021.12 |



## 2020

|                                官方库                                |                    项目注释                     |     备注     |  更新  |
| :------------------------------------------------------------------: | :---------------------------------------------: | :----------: | :----: |
|   [DynamicReLU(ECCV 2020)](https://github.com/Islanna/DynamicReLU)   | [注释](https://github.com/bobo0810/DynamicReLU) | 动态激活函数 | 2020.9 |
|     [AMSoftmax](https://github.com/cavalleria/cavaface.pytorch)      | [注释](https://github.com/bobo0810/FaceVerLoss) |  乘法角间隔  | 2020.9 |
| [ArcFace(CVPR 2019)](https://github.com/cavalleria/cavaface.pytorch) | [注释](https://github.com/bobo0810/FaceVerLoss) |  加法角间隔  | 2020.9 |
| [CircleLoss(CVPR 2020)](https://github.com/xialuxi/CircleLoss_Face)  | [注释](https://github.com/bobo0810/FaceVerLoss) |  加权角间隔  | 2020.9 |

|                                                                                      自研                                                                                      |                   备注                   | 更新    |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------: | ------- |
|           [FunnelRelu(ECCV 2020)](https://github.com/bobo0810/FunnelAct_Pytorch)![Github stars](https://img.shields.io/github/stars/bobo0810/FunnelAct_Pytorch.svg)            |            新型激活函数-复现             | 2020.7  |
|                       [AMP](https://github.com/bobo0810/PytorchExample) ![Github stars](https://img.shields.io/github/stars/bobo0810/PytorchExample.svg)                       |            自动混合精度-示例             | 2020.10 |
|                    [BroadFace(ECCV 2020)](https://github.com/bobo0810/BroadFace)![Github stars](https://img.shields.io/github/stars/bobo0810/BroadFace.svg)                    |        人脸对比队列更新策略-复现         | 2020.10 |
|                        [TargetDrop](https://github.com/bobo0810/TargetDrop)![Github stars](https://img.shields.io/github/stars/bobo0810/TargetDrop.svg)                        |          注意力机制Dropout-复现          | 2020.10 |
|                         [HS-ResNet](https://github.com/bobo0810/HS-ResNet)![Github stars](https://img.shields.io/github/stars/bobo0810/HS-ResNet.svg)                          |            ResNet改进版-复现             | 2020.11 |
| [AFF-ResNet(WACV 2021)](https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/README_CN.md)![Github stars](https://img.shields.io/github/stars/YimianDai/open-aff.svg) | 特征融合的统一方式- 复现，**并入官方库** | 2020.11 |
|                       [DDP](https://github.com/bobo0810/PytorchExample) ![Github stars](https://img.shields.io/github/stars/bobo0810/PytorchExample.svg)                       |           分布式数据并行-示例            | 2020.11 |
|                  [DiscFace(ACCV 2020)](https://github.com/bobo0810/FaceVerLoss)![Github stars](https://img.shields.io/github/stars/bobo0810/FaceVerLoss.svg)                   |            最小差异学习-复现             | 2020.12 |
|                        [NPCFace](https://github.com/bobo0810/FaceVerLoss)![Github stars](https://img.shields.io/github/stars/bobo0810/FaceVerLoss.svg)                         |            正负联合监督-复现             | 2020.12 |



## 2017-2019
|                                 官方库                                  |                                                                项目注释                                                                |       备注       |
| :---------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------: | :--------------: |
|     [Fatser Rcnn(NIPS 2015)](https://zhuanlan.zhihu.com/p/32404424)     |                            [注释](https://github.com/bobo0810/PytorchNetHub/tree/master/FasterRcnn_pytorch)                            |     目标检测     |
|   [YOLO v1(CVPR 2016)](https://github.com/xiongzihua/pytorch-YOLO-v1)   |                              [注释](https://github.com/bobo0810/PytorchNetHub/tree/master/Yolov1_pytorch)                              |     目标检测     |
| [YOLO v3(ECCV 2016)](https://github.com/eriklindernoren/PyTorch-YOLOv3) | [注释](https://github.com/bobo0810/PyTorch-YOLOv3-master) [重构](https://github.com/bobo0810/PytorchNetHub/tree/master/Yolov3_pytorch) |     目标检测     |
|         [YOLO v4](https://github.com/Tianxiaomo/pytorch-YOLOv4)         |                                           [注释](https://github.com/bobo0810/YOLOv4_Pytorch)                                           |     目标检测     |
|             [SSD](https://github.com/amdegroot/ssd.pytorch)             |       [注释](https://github.com/bobo0810/pytorchSSD)   [重构](https://github.com/bobo0810/PytorchNetHub/tree/master/SSD_pytorch)       |     目标检测     |
|         [CAM(CVPR 2016)](https://github.com/jacobgil/keras-cam)         |                               [注释](https://github.com/bobo0810/PytorchNetHub/tree/master/CAM_pytorch)                                |    特征可视化    |
|     [U-Net(MICCAI 2015)](https://github.com/milesial/Pytorch-UNet)      |                               [注释](https://github.com/bobo0810/PytorchNetHub/tree/master/UNet_pytorch)                               | 医学影像语义分割 |
|        [FPN(CVPR 2017)](https://github.com/kuangliu/pytorch-fpn)        |                               [注释](https://github.com/bobo0810/PytorchNetHub/tree/master/FPN_pytorch)                                |    特征金字塔    |

|                                                                     自研                                                                     | 备注                                           |
| :------------------------------------------------------------------------------------------------------------------------------------------: | ---------------------------------------------- |
| [XueLangTianchi](https://github.com/bobo0810/XueLangTianchi)![Github stars](https://img.shields.io/github/stars/bobo0810/XueLangTianchi.svg) | 雪浪制造AI挑战赛—视觉计算辅助良品检测 -32/2403 |
|       [S4ND](https://github.com/bobo0810/S4ND_Pytorch) ![Github stars](https://img.shields.io/github/stars/bobo0810/S4ND_Pytorch.svg)        | 单次单尺度肺结节检测（MICCAI 2018）复现        |
|    [3D SKconv](https://github.com/bobo0810/SKNet_Pytorch) ![Github stars](https://img.shields.io/github/stars/bobo0810/SKNet_Pytorch.svg)    | 注意力机制SE模块、SK模块的3D实现               |


> 注：猫狗分类、风格迁移、生成对抗等更多内容请访问[pytorch-book](https://github.com/chenyuntc/pytorch-book)





[![Star History Chart](https://api.star-history.com/svg?repos=bobo0810/PytorchNetHub&type=Date)](https://star-history.com/#bobo0810/PytorchNetHub&Date)
