

# 本仓库目的

- 如果不想阅读源码，只想傻瓜式跟着教程跑起来，推荐看原仓库地址，感谢大佬们的无私奉献！
- 本仓库将代码改为统一风格，更加规范化，能够更容易、更轻松得阅读源码，以便根据自己需求进行修改。

----------
# 接下来工作

### 目标检测
- [x] Faster rcnn
- [x] YOLO v1
- [x] YOLO v3
- [x] SSD
- [x] FP_SSD
- [x] Class Activation Mapping(CAM)
- [x] S4ND
- [ ] [YOLO v4](https://github.com/bobo0810/YOLOv4_Pytorch) (推理注释完毕，训练部分正在跟进)

### 语义分割
- [x] U-Net

### 其他
- [x] SKNet
- [x] FPN特征金字塔网络
- [x] Noise2noise
- [ ] 该仓库将拆分为多个子仓库，仅保留readme.md



 
----------
# 目标检测

### Fatser Rcnn
- 参考知乎[从编程实现角度学习FasterR-CNN](https://zhuanlan.zhihu.com/p/32404424)
- 基本理清结构，尚未完全理解

### YOLO v1

- [重构版本](https://github.com/bobo0810/AnnotatedNetworkModelGit/tree/master/Yolov1_pytorch) 强烈推荐！
- [原地址](https://github.com/xiongzihua/pytorch-YOLO-v1)

### YOLO v3

- [重构版本](https://github.com/bobo0810/AnnotatedNetworkModelGit/tree/master/Yolov3_pytorch) 强烈推荐！

- [原地址](https://github.com/eriklindernoren/PyTorch-YOLOv3)

- [原地址的加注释版本](https://github.com/bobo0810/PyTorch-YOLOv3-master) 

### YOLO v4

- [原地址](https://github.com/Tianxiaomo/pytorch-YOLOv4)

- [原地址的注释版本](https://github.com/bobo0810/YOLOv4_Pytorch) 

### SSD

- [重构版本](https://github.com/bobo0810/AnnotatedNetworkModelGit/tree/master/SSD_pytorch) 强烈推荐！
- [原地址](https://github.com/amdegroot/ssd.pytorch) 
- [原地址的加注释版本](https://github.com/bobo0810/pytorchSSD) 

### FP_SSD

- [实现](https://github.com/bobo0810/AnnotatedNetworkModelGit/tree/master/FP_SSD_pytorch)
- 论文地址:[基于特征金字塔的 SSD 目标检测改进算法](https://pan.baidu.com/s/1oXYksRiqvtN-LCAdcYfEIg)

### Class Activation Mapping
- 作用：分类、定位（不使用真值框进行定位，论文证明 卷积层本身就有定位功能）
- 实现：[CAM_pytorch](https://github.com/bobo0810/AnnotatedNetworkModelGit/tree/master/CAM_pytorch)
- 论文地址:[CVPR 2016  Learning Deep Features for Discriminative Localization](https://arxiv.org/pdf/1512.04150.pdf)


### S4ND
- 作用：单次单尺度肺结节检测
- 实现：[S4ND_Pytorch](https://github.com/bobo0810/S4ND_Pytorch) 
- 论文地址:[MICCAI2018 S4ND](https://arxiv.org/pdf/1805.02279.pdf?fbclid=IwAR0B3dI8tjvWz-Mk9Xpyymfnk-SNs6k8tw2B8HU3dTTP-vFinQURHGZSCQs)

----------
# 语义分割

### U-Net
- [原地址](https://github.com/milesial/Pytorch-UNet)
- [原地址的加注释版本](https://github.com/bobo0810/AnnotatedNetworkModelGit/tree/master/UNet_pytorch) 

----------

# 其他


### SKNet
- 作用：继Res\Dense\SE Block之后新的Block
- 实现：[SKNet](https://github.com/bobo0810/SKNet_Pytorch)
- 论文地址:[CVPR 2019 SKNet](https://arxiv.org/abs/1903.06586)


### 图像去噪

 - 实现：[ImageDenoising_pytorch](https://github.com/bobo0810/AnnotatedNetworkModelGit/tree/master/ImageDenoising_pytorch)
 - 论文: 基于深度卷积神经网络的图像去噪研究


### Noise2noise
- [重构版本](https://github.com/bobo0810/AnnotatedNetworkModelGit/tree/master/Noise2noise_pytorch)
- [原地址](https://github.com/joeylitalien/noise2noise-pytorch) 


### FPN特征金字塔网络
- 实现：[FPN_pytorch](https://github.com/bobo0810/AnnotatedNetworkModelGit/tree/master/FPN_pytorch)
- [原地址](https://github.com/kuangliu/pytorch-fpn) 


注：猫狗大战、风格迁移、GAN生成对抗网络在[pytorch-book传送门](https://github.com/chenyuntc/pytorch-book),更多内容请进门访问，感谢大佬无私奉献。

----------

# 阅读源码三步骤

- 数据预处理
- 网络模型搭建
- 损失函数定义


----------


# 项目一般结构


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

