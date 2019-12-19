# U-Net网络

----------

该仓库基于[milesial](https://github.com/milesial)的[Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)进行的，非常感谢大佬无私的奉献。
 

- [原地址](https://github.com/milesial/Pytorch-UNet) 
- [原地址的加注释版本](https://github.com/bobo0810/AnnotatedNetworkModelGit/tree/master/UNet_pytorch) 

----------

 # 目前支持：

- 数据集： Kaggle's [Carvana Image Masking Challenge](https://pan.baidu.com/s/1tQI7aQ4y9k0K3qBjCnJ53Q)
- 网络：U-Net


 # 相比原作者的特点：

- 所有参数均可在config.py中设置
- 重新整理结构，并加入大量代码注释
- loading

----------

 - 环境：

    | python版本  |  pytorch版本 |
    | ----------- | ----------   |
    |  3.5  | 0.4   |
 
 - 依赖：

       pip install pydensecrf
      
----------

# U-Net网络结构

![](https://github.com/bobo0810/imageRepo/blob/master/img/659347.jpg)

- ###### 原论文左侧 conv 3x3 无pad，故每次conv后feature map尺寸缩小。故与右侧feature map融合之前需要裁剪。
- ###### 该仓库左侧 conv 3x3 pad=1，故每次conv后feature map尺寸不变。故反卷积后保证尺度统一与右侧feature map融合即可。
  

----------

# 准备数据集：
下载Kaggle's [Carvana Image Masking Challenge](https://pan.baidu.com/s/1tQI7aQ4y9k0K3qBjCnJ53Q)数据集，并在utils/config.py中配置数据集的根目录。
```
CarvanaImageMaskingChallenge
│
└───train
│   │   xxx.gif
│   │   ...
│   
└───train_masks
│   │   xxx.jpg
│   │   ...
```


----------

# Trian:

1、在config.py中配置训练参数

2、执行train.py开始训练

----------

# Eval:

每训练一轮epoch都将计算Dice距离（用于度量两个集合的相似性）
----------

# Predict:

功能：可视化一张预测图片

1、将预训练模型放到项目根目录下

预训练模型下载：[MODEL.pth](https://pan.baidu.com/s/1D_OtX16iL3aJefvOqyRWnw)

2、预测单张图片

        python predict.py -i image.jpg -o output.jpg

3、预测多张图片并显示

        python predict.py -i image1.jpg image2.jpg --viz --no-save


<div align="center">
<img src="https://github.com/bobo0810/imageRepo/blob/master/img/78620180.jpg" width="400px"  height="300px" alt="图片说明" ><img src="https://github.com/bobo0810/imageRepo/blob/master/img/22328540.jpg" width="400px"  height="300px" alt="图片说明" > 
</div>


----------

# 关于作者

- 原作者 [milesial](https://github.com/milesial)

- 本仓库作者 [Mr.Li](https://github.com/bobo0810)