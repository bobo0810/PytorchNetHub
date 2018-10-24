# 代码注释

加入大量代码注释， 以便理解

update:2018年10月24日16:09:31

----------

# 阅读源码三步骤

- 数据预处理
- 网络模型搭建
- 损失函数定义

----------
# 接下来工作
 
- Faster rcnn  源码注释（趁早填坑）
- ~~FPN特征金字塔网络 源码注释~~
- ~~SSD 源码注释并重构~~
- ~~YOLO v1  源码注释并重构~~
- ~~YOLO v3  源码注释并重构~~  
- ~~U-Net 源码注释~~
- ~~FP_SSD~~
- Mask rcnn（趁早填坑）
- ~~Class Activation Mapping(CAM 类激活映射)~~

----------
# 目标检测网络

### Fatser Rcnn
- 参考知乎[从编程实现角度学习FasterR-CNN](https://zhuanlan.zhihu.com/p/32404424)
- 基本理清结构，尚未完全理解

### YOLO V1

- [重构版本](https://github.com/bobo0810/AnnotatedNetworkModelGit/tree/master/Yolov1_pytorch) 强烈推荐！
- [原地址](https://github.com/xiongzihua/pytorch-YOLO-v1)

### YOLO V3

- [重构版本](https://github.com/bobo0810/AnnotatedNetworkModelGit/tree/master/Yolov3_pytorch) 强烈推荐！

- [原地址](https://github.com/eriklindernoren/PyTorch-YOLOv3)

- [原地址的加注释版本](https://github.com/bobo0810/PyTorch-YOLOv3-master) 


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

----------
# 目标分割网络

### U-Net
- [原地址](https://github.com/milesial/Pytorch-UNet)
- [原地址的加注释版本](https://github.com/bobo0810/AnnotatedNetworkModelGit/tree/master/UNet_pytorch) 

----------

# 其他


### 图像去噪论文复现

 - 实现：[papersReproduced](https://github.com/bobo0810/AnnotatedNetworkModelGit/tree/master/ImageDenoising_pytorch)
 - 论文地址: [基于深度卷积神经网络的图像去噪研究](http://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CJFQ&amp;dbname=CJFDLAST2017&amp;filename=JSJC201703042&amp;uid=WEEvREcwSlJHSldRa1FhdXNXa0hIb3VVSnliNDU0a2dObEJYUVM1MzR2cz0=$9A4hF_YAuvQ5obgVAqNKPCYcEjKensW4ggI8Fm4gTkoUKaID8j8gFw!!&amp;v=MTUzMzkxRnJDVVJMS2ZZdWRvRnk3blVydkJMejdCYmJHNEg5Yk1ySTlCWm9SOGVYMUx1eFlTN0RoMVQzcVRyV00=)

### FPN特征金字塔网络
- 实现：[pytorch-FPN](https://github.com/bobo0810/AnnotatedNetworkModelGit/tree/master/FPN_pytorch)
- [原地址](https://github.com/kuangliu/pytorch-fpn) 


注：猫狗大战、风格迁移、GAN生成对抗网络在[pytorch-book传送门](https://github.com/chenyuntc/pytorch-book),更多内容请进门访问，感谢大佬无私奉献。


----------


# 项目一般结构


- 总结构

  ![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-26/99053959.jpg)
  
  
- 项目结构

  1、定义网络
  
  ![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-26/16409622.jpg) 
  
   2、封装数据集
   
  ![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-26/38894621.jpg)
  
   3、工具类
   
  ![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-26/98583532.jpg)
  
   4、主函数
   
  ![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-26/32257225.jpg)
  

