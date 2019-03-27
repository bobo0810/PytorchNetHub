# Noise2Noise: Learning Image Restoration without Clean Data

## 依赖

* Ubuntu
* Python(3.6.8)
* PyTorch(1.0.1)
* Torchvision(0.2.2)
* NumPy(1.16.2)
* Matplotlib(3.0.3)
* Pillow(5.4.1)
* ~~OpenEXR(1.3.0，仅蒙特卡洛图像用到，去噪和去水印均不受影响，不易安装，本仓库注释掉相关代码)~~

## 改动项

- 规范化为统一风格
- 所有参数在config.py中配置
- 新增visdom可视化训练过程(暂时未加)



## 数据集

任何数据集都可以，[COCO 2017](http://cocodataset.org/#download) (1 GB) 比较容易去训练/验证。下面代码将获得训练集/验证集：4200/800 
```
mkdir data && cd data
mkdir train valid test
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip && cd val2017
mv `ls | head -4200` ../train
mv `ls | head -800` ../valid
```

## Training
main.py

## 论文
* Jaakko Lehtinen, Jacob Munkberg, Jon Hasselgren, Samuli Laine, Tero Karras, Miika Aittala,and Timo Aila. [*Noise2Noise: Learning Image Restoration without Clean Data*](https://research.nvidia.com/publication/2018-07_Noise2Noise%3A-Learning-Image). Proceedings of the 35th International Conference on Machine Learning, 2018.