 - 环境：

    | python版本  |  pytorch版本 |
    | ----------- | ----------   |
    |  3.5  | 0.3.0   |

 - 说明：

   1、基本实现参考： [pytorchYOLOv1master][1]

   2、仅重构代码，并未提升效果

   3、 测试时，在VOC2012训练集上loss为0.1左右，在VOC2007测试集上loss基本降低   很少。怀疑过拟合。

 - 当前工作：
   
   1、~~训练完，可视化测试图像和loss等~~

   2、~~将训练好的模型放到这里~~
   
   3、~~准备添加注释，以便理解~~
   
   4、尝试优化网络模型，提高mAP
   
 - 改进方向：

   1、更改学习率

   2、~~ 调整网络结构（参考版本为vgg16，试试残差）~~

   3、~~ 更改优化器从SGD到Adam ~~

- 下载网络模型：

  1、在VOC2007测试集上验证的效果最好的一个网络模型（[百度网盘](https://pan.baidu.com/s/1HCO24KGqjJw01raiCB7f2A)）

  2、保存的最后一个网络模型 （[百度网盘](https://pan.baidu.com/s/1HKY7qGgK7i3Fv_ks9ldflw)）

- 效果:

   验证集:voc2012训练集
   
   模型:在VOC2007测试集上验证的效果最好的一个网络模型

![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-24/60726940.jpg)



### loss趋势

| epoch   | VOC2007测试集的loss  |
| --- | ------------------ |
| 0   | 5.806424896178707  |
| 1   | 5.855176733386132  |
| 2   | 5.9203009036279495 |
| ... |     ...               |
| 118 | 5.187265388427242  |
| 119 | 5.190768877152474  |

![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-25/15532458.jpg)

 注：蓝线为在VOC2012训练集上的loss，黄线为 VOC2007测试集的loss
 
### 网络表现

- 最后保存的模型 在VOC2007验证集的表现
<div align="center">
<img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-23/60783746.jpg" height="300px" alt="图片说明" ><img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-23/77507684.jpg" height="300px" alt="图片说明" > 
</div>
<div align="center">
<img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-23/67512569.jpg" height="300px" alt="图片说明" ><img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-23/71549266.jpg" height="300px" alt="图片说明" > 
</div>
<div align="center">
<img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-23/62106896.jpg" height="300px" alt="图片说明" ><img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-23/409753.jpg" height="300px" alt="图片说明" > 
</div>


- 最后保存的模型 在VOC2012训练集的表现（可能过拟合，在训练集表现优秀）

<div align="center">
<img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-23/10450103.jpg" height="300px" alt="图片说明" ><img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-23/91121778.jpg" height="300px" alt="图片说明" > 
</div>
<div align="center">
<img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-23/4137984.jpg" height="300px" alt="图片说明" > 
<img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-23/77285051.jpg" height="300px" alt="图片说明" > 
</div>
<div align="center">
<img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-23/25022049.jpg" height="300px" alt="图片说明" ><img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-23/49753476.jpg" height="300px" alt="图片说明" > 
</div>


[1]: https://github.com/xiongzihua/pytorch-YOLO-v1

#  以下为本人新增内容

- 新增内容：

   新增Resnet152网络来替换原作者的VGG16。（代码包括main_resnet.py 、models/resnet.py ）
   
- 实现细节：

   仅仅将Resnet152网络的最后一层全连接层的输出改为1470，再改变形状为7x7x30。
   
- 效果：

  极其不好，猜测原因为Resnet152网络是用来分类，将其直接用于回归导致效果不好。

- loss图：

| Resnet152+Ada优化器   | Resnet152+SGD优化器 |
| --- | ------------------ |
| 见左下 | 见右下  |

<div align="center">
<img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-25/94530436.jpg" height="300px" alt="图片说明" ><img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-25/24366463.jpg" height="300px" alt="图片说明" > 
</div>

 - 优化建议：
   
   1、~~使用Resnet50,网络最后处理参考原文VGG16处理试试~~
   
   2、Resnet50去掉最后一层，加入类似VGG16的两层全连接层+Drop等 。loss仍降不下去，该项目以后不再提供提升效果的内容。


 - 特别鸣谢：朱辉师兄

   



