# 重构YOLO v3代码实现

----------

该仓库基于[eriklindernoren](https://github.com/eriklindernoren)的[PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)进行的，非常感谢他无私的奉献。
 

- [原地址](https://github.com/eriklindernoren/PyTorch-YOLOv3) 
- [原地址的加注释版本](https://github.com/bobo0810/PyTorch-YOLOv3-master) 
- [重构版本](https://github.com/bobo0810/AnnotatedNetworkModelGit/tree/master/Yolov3_pytorch) 强烈推荐！（即本仓库）

----------

 # 目前支持：

- 数据集：COCO
- 网络：Darknet-52

 # 相比原作者的特点：

- 所有参数均可在config.py中设置
- 重新整理结构，并加入大量代码注释
- 加入visdom可视化


----------

# 一般项目结构

  1、定义网络
  
  ![](https://github.com/bobo0810/imageRepo/blob/master/img/16409622.jpg) 
  
   2、封装数据集
   
  ![](https://github.com/bobo0810/imageRepo/blob/master/img/38894621.jpg)
  
   3、工具类
   
  ![](https://github.com/bobo0810/imageRepo/blob/master/img/98583532.jpg)
  
   4、主函数
   
  ![](https://github.com/bobo0810/imageRepo/blob/master/img/32257225.jpg)

- 环境：

    | python版本  |  pytorch版本 |
    | ----------- | ----------   |
    |  3.5  | 0.4   |

----------

# Darknet-52网络结构

![](https://github.com/bobo0810/imageRepo/blob/master/img/16734558.jpg)

以下阅读源码有用：

hyperparams

![](https://github.com/bobo0810/imageRepo/blob/master/img/97781689.jpg)

module_list

![](https://github.com/bobo0810/imageRepo/blob/master/img/10165593.jpg)

module_defs

![](https://github.com/bobo0810/imageRepo/blob/master/img/56737437.jpg)



----------

# 准备数据集：
下载COCO的数据集

```
$ cd data/
$ bash get_coco_dataset.sh
```

数据集结构
```
data/coco
│
└───images
│   │   train2014
│   │   val2014
│   
└───labels
│   │   train2014
│   │   val2014
│   ...
│   ...

```

----------

# Trian:

1、开启Visdom（类似TnsorFlow的tensorboard,可视化工具）

```
# First install Python server and client
pip install visdom
# Start the server 
python -m visdom.server
```

2、开始训练

在config.py中设置参数。

在main.py中将运行train()。

###### 由于原仓库保存、加载模型bug，故不支持保存为 .weight官方格式（二进制且仅保存conv和bn层参数，其余参数读取cfg文件即可），训练保存模型为.pt模型（保存整个模型）。

![](https://github.com/bobo0810/imageRepo/blob/master/img/68971633.jpg)

----------

# Test:

作用：测试，计算mAP

1、下载官方的预训练模型

```
$ cd checkpoints/
$ bash download_weights.sh
```

2、在config.py中load_model_path配置预训练模型的路径

###### 支持官方模型 .weight 和 自训练模型 .pt 

3、 在config.py中设置参数。
   
    在main.py中运行test()。


| Model               | mAP (min. 50 IoU) |
|---------------------|-------------------|
| YOLOv3 (paper)      | 57.9              |
| YOLOv3 (官方)       | 58.38             |
| YOLOv3 (this impl.) | 58.2              |



![](https://github.com/bobo0810/imageRepo/blob/master/img/77791130.jpg)

----------

# Predict:

功能：可视化预测图片

1、在config.py中load_model_path配置预训练模型的路径
   ###### 支持官方模型 .weight 和 自训练模型 .pt 
2、在config.py中设置参数。
   
   在main.py中将运行detect()。


官方模型效果：

<div align="center">
<img src="https://github.com/bobo0810/imageRepo/blob/master/img/11371083.jpg" width="400px"  height="300px" alt="图片说明" > 
<img src="https://github.com/bobo0810/imageRepo/blob/master/img/39079856.jpg" width="400px"  height="300px" alt="图片说明" > 
</div>
<div align="center">
<img src="https://github.com/bobo0810/imageRepo/blob/master/img/91451324.jpg" width="400px"  height="300px" alt="图片说明" ><img src="https://github.com/bobo0810/imageRepo/blob/master/img/28759426.jpg" width="400px"  height="300px" alt="图片说明" > 
</div>





----------

## 参考文献：

推荐配合阅读，效果更佳~

- [从0到1实现YOLOv3（part one）](https://blog.csdn.net/qq_25737169/article/details/80530579)

- [从0到1实现YOLO v3（part two）](https://blog.csdn.net/qq_25737169/article/details/80634360)

- [yolo v3 译文](https://zhuanlan.zhihu.com/p/34945787)

- [YOLO v3网络结构分析](https://blog.csdn.net/qq_37541097/article/details/81214953)

----------

# 关于作者

- 原作者 [eriklindernoren](https://github.com/eriklindernoren)

- 本仓库作者 [Mr.Li](https://github.com/bobo0810)



