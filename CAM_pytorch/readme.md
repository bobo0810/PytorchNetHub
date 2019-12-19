#  class activation mapping


 - 环境：

    | python版本  |  pytorch版本 |
    | ----------- | ----------   |
    |  3.5  | 0.3.0   |


- 作用：分类、定位（不使用真值框进行定位，论文证明 卷积层本身就有定位功能）

----------

## 数据集

 - [INRIA Person数据集(官方)](http://pascal.inrialpes.fr/data/human/)    
 - [INRIA Person数据集(百度云)](https://pan.baidu.com/s/1adTzYgX13K4CIjZNODRXqQ)


## 预训练模型

- [VGG16_CAM_39_99.455.pth](https://pan.baidu.com/s/1OVnxBBhmtVgTEUz0nNmrFg)
    

## 训练

1、在config.py中配置数据集等训练参数

2、执行main.py开始训练

## 可视化

1、在config.py中配置预训练模型

2、执行main.py可视化class_activation_map




## 训练过程
<div align="center">
<img src="https://github.com/bobo0810/imageRepo/blob/master/img/15578714.jpg" width="400px"  height="300px" alt="图片说明" ><img src="https://github.com/bobo0810/imageRepo/blob/master/img/18-10-19/81997632.jpg" width="400px"  height="300px" alt="图片说明" > 
</div>

----------

## 效果

- 网络分类时重点关注的区域(即网络的分类依据)

<div align="center">
<img src="https://github.com/bobo0810/imageRepo/blob/master/img/97478889.jpg" width="400px"  height="300px" alt="图片说明" ><img src="https://github.com/bobo0810/imageRepo/blob/master/img/91606455.jpg" width="400px"  height="300px" alt="图片说明" > 
</div>

<div align="center">
<img src="https://github.com/bobo0810/imageRepo/blob/master/img/94097073.jpg" width="400px"  height="300px" alt="图片说明" ><img src="https://github.com/bobo0810/imageRepo/blob/master/img/27379841.jpg" width="400px"  height="300px" alt="图片说明" > 
</div>

<div align="center">
<img src="https://github.com/bobo0810/imageRepo/blob/master/img/32549346.jpg" width="400px"  height="300px" alt="图片说明" ><img src="https://github.com/bobo0810/imageRepo/blob/master/img/53559366.jpg" width="400px"  height="300px" alt="图片说明" > 
</div>

----------

## 参考

- [Keras implementation of CAM](https://github.com/jacobgil/keras-cam)
- [可视化CNN](https://github.com/huanghao-code/VisCNN_CVPR_2016_Loc)
- [论文CVPR 2016](https://arxiv.org/pdf/1512.04150.pdf)