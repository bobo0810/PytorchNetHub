# 重构SSD代码实现

该仓库基于[Max deGroot](https://github.com/amdegroot)与[Ellis Brown](https://github.com/ellisbrown)的[ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)进行的，非常感谢他们无私的奉献。
 

 # 目前支持：

原作者支持VOC、COCO，该仓库仅支持VOC，如果有时间，考虑将COCO加上。

# 原因：

 大牛们写代码果然不拘小节，结构混乱依然不影响他们这么优秀。强迫症犯了，一周时间理解源码，一天内重构完成。哇，世界清爽了~

 ###### 注：该项目功能上并未进行任何修改，仅做重构，用于理解。


# 相比原作者的特点：
- 所有参数均可在config.py中设置
- 重新整理结构，并加入大量代码注释

# 准备数据集：
下载VOC2007和VOC2012的数据集，并在utils/config.py中的voc_data_root配置数据集的根目录。
```
VOCdevkit
│
└───VOC2007
│   │   JPEGImages
│   │   ImageSets
│   │   Annotations
│   │   ...
│   
└───VOC2012
│   │   JPEGImages
│   │   ImageSets
│   │   Annotations
│   │   ...
```

# Trian:

作用：使用VOC2007和2012的训练集+验证集 开始训练

1、开启Visdom（类似TnsorFlow的tensorboard,可视化工具）
```
# First install Python server and client
pip install visdom
# Start the server 
python -m visdom.server
```
2、下载SSD的基础网络VGG16(去掉fc层)

下载地址：[vgg16_reducedfc.pth](https://pan.baidu.com/s/19Iumt072GMiFGlS5lVNy1Q)

下载完成后将其放置在checkpoint文件夹下即可。也可通过配置config.py中basenet的路径。

3、开始训练

在main.py中将train()注释取消，其他方法注释掉，即可运行。

# Eval:

作用：VOC2007测试集,计算各类AP及mAP

1、在config.py中load_model_path配置预训练模型的路径

预训练模型下载：[ssd300_VOC_100000.pth](https://pan.baidu.com/s/1hrJo__owbF3ufepwJJ0uzA)


2、在main.py中将eval()注释取消，其他方法注释掉，即可运行。

# Test:

功能：VOC2007测试集，将预测结果写入txt

1、在config.py中load_model_path配置预训练模型的路径

预训练模型下载：[ssd300_VOC_100000.pth](https://pan.baidu.com/s/1hrJo__owbF3ufepwJJ0uzA)

2、在main.py中将test()注释取消，其他方法注释掉，即可运行。

结果：
![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-6-8/68841398.jpg)

# Predict:

功能：可视化一张预测图片

1、在config.py中load_model_path配置预训练模型的路径

预训练模型下载：[ssd300_VOC_100000.pth](https://pan.baidu.com/s/1hrJo__owbF3ufepwJJ0uzA)
2、在main.py中将predict()注释取消，其他方法注释掉，即可运行。

结果：

原图:
![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-7-25/88568756.jpg)

预处理之后的图像:
![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-7-25/33336596.jpg)

结果:
![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-7-25/48601434.jpg)

# 关于作者

- 原作者 [Max deGroot](https://github.com/amdegroot)、[Ellis Brown](https://github.com/ellisbrown)

- 本仓库作者 [Mr.Li](https://github.com/bobo0810)