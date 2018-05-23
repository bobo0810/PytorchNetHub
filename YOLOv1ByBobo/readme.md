---
title: YOLOv1ByBobo
tags: yolo,pytorch

---

 - 环境：

    | python版本  |  pytorch版本 |
    | ----------- | ----------   |
    |  3.5  | 0.3.0   |  

 - 说明：
 
   1、仅重构代码，并未提升效果
 - 当前工作：
    
   1、~~训练完，可视化测试图像和loss等~~

   2、~~将训练好的模型放到这里~~
   
   3、测试时，在VOC2012训练集上loss为0.1左右，在VOC2007测试集上loss基本降低很少。怀疑学习率设置问题或者网络结构问题，之后更改学习率，测试第二版。
 
- 下载网络模型：

  1、在VOC2007测试集上验证的效果最好的一个网络模型（[百度网盘](https://pan.baidu.com/s/1HCO24KGqjJw01raiCB7f2A)）

  2、保存的最后一个网络模型 （[百度网盘](https://pan.baidu.com/s/1HKY7qGgK7i3Fv_ks9ldflw)）



### loss趋势

| epoch   | VOC2007测试集的loss  |
| --- | ------------------ |
| 0   | 5.806424896178707  |
| 1   | 5.855176733386132  |
| 2   | 5.9203009036279495 |
| ... |     ...               |
| 118 | 5.187265388427242  |
| 119 | 5.190768877152474  |

### 网络表现

- 最后保存的模型 在VOC2007验证集的表现
<div align="center">
<img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-23/60783746.jpg" height="300px" alt="图片说明" ><img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-23/77507684.jpg" height="300px" alt="图片说明" > 
<img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-23/67512569.jpg" height="300px" alt="图片说明" ><img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-23/71549266.jpg" height="300px" alt="图片说明" > 
<img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-23/62106896.jpg" height="300px" alt="图片说明" ><img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-23/409753.jpg" height="300px" alt="图片说明" > 
</div>


- 最后保存的模型 在VOC2012训练集的表现（可能过拟合，在训练集表现优秀）

<div align="center">
<img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-23/10450103.jpg" height="300px" alt="图片说明" ><img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-23/91121778.jpg" height="300px" alt="图片说明" > 
<img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-23/4137984.jpg" height="300px" alt="图片说明" ><img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-23/77285051.jpg" height="300px" alt="图片说明" > 
<img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-23/25022049.jpg" height="300px" alt="图片说明" ><img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-23/49753476.jpg" height="300px" alt="图片说明" > 
</div>