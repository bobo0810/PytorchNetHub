

# 目的

- 论文复现

- 算法竞赛

- 项目注释 

  > 即开即用推荐官方库,阅读源码推荐注释
  


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

### 主干网络

- [x] FPN
- [x] SKNet
- [x] HS-ResNet
- [x] AFF-ResNet [并入官方库](https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/README_CN.md)
- [x] TargetDrop

### 激活函数

- [x] FunnelAct
- [x] DynamicReLU

### 人脸识别

- [x] Semi-Siamese-Training
- [x] BroadFace

### 人脸损失

- [x] AMSoftmax
- [x] ArcFace
- [x] Circle-Loss
- [x] DiscFace
- [x] NPCFace

### 知识蒸馏

- [x] RepDistiller

### 算法竞赛

- [x] [雪浪制造AI挑战赛](https://github.com/bobo0810/XueLangTianchi)    排名: 32/2403
- [ ] [计图”人工智能算法挑战赛-狗细分类]()   待比赛结束开源

### 其他
- [x] AMP - 自动混合精度
- [x] DDP - 分布式数据并行



# 汇总

## 2021

|                  模型                   |                         官方库                         |                           注释                            | 实现 |   来源    |     备注     |  更新  |
| :-------------------------------------: | :----------------------------------------------------: | :-------------------------------------------------------: | :--: | :-------: | :----------: | :----: |
| [SST](https://arxiv.org/abs/2007.08398) | [Link](https://github.com/dituu/Semi-Siamese-Training) | [注释](https://github.com/bobo0810/Semi-Siamese-Training) |      | ECCV 2020 | 浅层人脸学习 | 2021.2 |
|              RepDistiller               |   [Link](https://github.com/HobbitLong/RepDistiller)   |     [注释](https://github.com/bobo0810/RepDistiller)      |      |           |  知识蒸馏库  | 2021.2 |



## 2020
|模型|官方库|注释|实现|来源|备注|更新|
|:---:|:----:|:---:|:------:|:------:|:------:|:------:|
|[FunnelRelu](https://arxiv.org/pdf/2007.11824.pdf)|[MegEngine](https://github.com/megvii-model/FunnelAct) ||[复现](https://github.com/bobo0810/FunnelAct_Pytorch)|ECCV 2020|较PRelu等更有效|2020.7|
|[DynamicReLU](https://arxiv.org/abs/2003.10027)|[Link](https://github.com/Islanna/DynamicReLU)|[注释](https://github.com/bobo0810/DynamicReLU)||ECCV 2020|动态激活函数|2020.9|
|[AMSoftmax](https://arxiv.org/pdf/1801.05599.pdf)|[Link](https://github.com/cavalleria/cavaface.pytorch)|[注释](https://github.com/bobo0810/FaceVerLoss)|||乘法角间隔|2020.9|
|[ArcFace](https://arxiv.org/abs/1801.07698)|[Link](https://github.com/cavalleria/cavaface.pytorch)|[注释](https://github.com/bobo0810/FaceVerLoss)||CVPR 2019|加法角间隔|2020.9|
|[CircleLoss](https://arxiv.org/abs/2002.10857)|[Link](https://github.com/xialuxi/CircleLoss_Face)|[注释](https://github.com/bobo0810/FaceVerLoss)||CVPR 2020|加权角间隔|2020.9|
|AMP|||[实现](https://github.com/bobo0810/PytorchNetHub/tree/master/AMP)||自动混合精度|2020.10|
|[BroadFace](https://arxiv.org/abs/2008.06674)|||[复现](https://github.com/bobo0810/BroadFace)|ECCV 2020|队列更新|2020.10|
|[TargetDrop](https://arxiv.org/abs/2010.10716)|||[复现](https://github.com/bobo0810/TargetDrop)||注意力Drop|2020.10|
|[HS-ResNet](https://arxiv.org/abs/2010.07621)|||[复现](https://github.com/bobo0810/HS-ResNet)||改进ResNet|2020.11|
|[AFF-ResNet](https://arxiv.org/abs/2009.14082)|[MXNet](https://github.com/YimianDai/open-aff)||[复现](https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/README_CN.md)|WACV 2021|统一特征融合|2020.11|
|DDP|||[实现](https://github.com/bobo0810/PytorchNetHub/tree/master/DDP)||分布式数据并行|2020.11|
|[DiscFace](https://openaccess.thecvf.com/content/ACCV2020/html/Kim_DiscFace_Minimum_Discrepancy_Learning_for_Deep_Face_Recognition_ACCV_2020_paper.html)|||[复现](https://github.com/bobo0810/FaceVerLoss)|ACCV 2020|最小差异学习|2020.12|
|[NPCFace](https://arxiv.org/abs/2007.10172)|||[复现](https://github.com/bobo0810/FaceVerLoss)||正负联合监督|2020.12|



## 2017-2019
|模型|官方库|注释|实现|来源|
|:---:|:----:|:---:|:------:|:------:|
|[Fatser Rcnn](https://arxiv.org/abs/1506.01497) |[Link](https://zhuanlan.zhihu.com/p/32404424)|[注释](https://github.com/bobo0810/PytorchNetHub/tree/master/FasterRcnn_pytorch)||NIPS 2015|
|[YOLO v1](https://arxiv.org/abs/1506.02640) |[Link](https://github.com/xiongzihua/pytorch-YOLO-v1)|[注释](https://github.com/bobo0810/PytorchNetHub/tree/master/Yolov1_pytorch)||CVPR 2016|
|[YOLO v3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) |[Link](https://github.com/eriklindernoren/PyTorch-YOLOv3)|[注释](https://github.com/bobo0810/PyTorch-YOLOv3-master)|[重构](https://github.com/bobo0810/PytorchNetHub/tree/master/Yolov3_pytorch)||
|[YOLO v4](https://arxiv.org/pdf/2004.10934.pdf) |[Link](https://github.com/Tianxiaomo/pytorch-YOLOv4)|[注释](https://github.com/bobo0810/YOLOv4_Pytorch)|||
|[SSD](https://arxiv.org/abs/1512.02325)|[Link](https://github.com/amdegroot/ssd.pytorch)|[注释](https://github.com/bobo0810/pytorchSSD)|[重构](https://github.com/bobo0810/PytorchNetHub/tree/master/SSD_pytorch)|ECCV 2016|
|[CAM](https://arxiv.org/pdf/1512.04150.pdf) |[Link](https://github.com/jacobgil/keras-cam)|[注释](https://github.com/bobo0810/PytorchNetHub/tree/master/CAM_pytorch)||CVPR 2016|
|[S4ND](https://arxiv.org/pdf/1805.02279.pdf?fbclid=IwAR0B3dI8tjvWz-Mk9Xpyymfnk-SNs6k8tw2B8HU3dTTP-vFinQURHGZSCQs) |||[复现](https://github.com/bobo0810/S4ND_Pytorch)|MICCAI 2018|
|[U-Net](https://arxiv.org/abs/1505.04597)|[Link](https://github.com/milesial/Pytorch-UNet)|[注释](https://github.com/bobo0810/PytorchNetHub/tree/master/UNet_pytorch) ||MICCAI 2015|
|[SKNet](https://arxiv.org/pdf/1903.06586.pdf)|[Link](https://github.com/implus/SKNet)||[实现](https://github.com/bobo0810/SKNet_Pytorch)|CVPR 2019|
|[FPN](https://arxiv.org/abs/1612.03144)|[Link](https://github.com/kuangliu/pytorch-fpn)|[注释](https://github.com/bobo0810/PytorchNetHub/tree/master/FPN_pytorch)||CVPR 2017|
|[XueLangTianchi](https://tianchi.aliyun.com/competition/entrance/231666/introduction)|||[实现](https://github.com/bobo0810/XueLangTianchi)|雪浪制造AI挑战赛|


> 注：猫狗分类、风格迁移、生成对抗等更多内容请访问[pytorch-book](https://github.com/chenyuntc/pytorch-book)


