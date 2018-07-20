# 注：本人新增内容在最下面

# SSD: Single Shot MultiBox Object Detector, in PyTorch
A [PyTorch](http://pytorch.org/) implementation of [Single Shot MultiBox Detector](http://arxiv.org/abs/1512.02325) from the 2016 paper by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang, and Alexander C. Berg.  The official and original Caffe code can be found [here](https://github.com/weiliu89/caffe/tree/ssd).


<img align="right" src= "https://github.com/amdegroot/ssd.pytorch/blob/master/doc/ssd.png" height = 400/>

### Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training-ssd'>Train</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#performance'>Performance</a>
- <a href='#demos'>Demos</a>
- <a href='#todo'>Future Work</a>
- <a href='#references'>Reference</a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Installation
- Install [PyTorch](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.
- Clone this repository.
  * Note: We currently only support Python 3+.
- Then download the dataset by following the [instructions](#datasets) below.
- We now support [Visdom](https://github.com/facebookresearch/visdom) for real-time loss visualization during training!
  * To use Visdom in the browser:
  ```Shell
  # First install Python server and client
  pip install visdom
  # Start the server (probably in a screen or tmux)
  python -m visdom.server
  ```
  * Then (during training) navigate to http://localhost:8097/ (see the Train section below for training details).
- Note: For training, we currently support [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [COCO](http://mscoco.org/), and aim to add [ImageNet](http://www.image-net.org/) support soon.

## Datasets
To make things easy, we provide bash scripts to handle the dataset downloads and setup for you.  We also provide simple dataset loaders that inherit `torch.utils.data.Dataset`, making them fully compatible with the `torchvision.datasets` [API](http://pytorch.org/docs/torchvision/datasets.html).


### COCO
Microsoft COCO: Common Objects in Context

##### Download COCO 2014
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/COCO2014.sh
```

### VOC Dataset
PASCAL VOC: Visual Object Classes

##### Download VOC2007 trainval & test
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

##### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

## Training SSD
- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at:              https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
- By default, we assume you have downloaded the file in the `ssd.pytorch/weights` dir:

```Shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

- To train SSD using the train script simply specify the parameters listed in `train.py` as a flag or manually change them.

```Shell
python train.py
```

- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.
  * For instructions on Visdom usage/installation, see the <a href='#installation'>Installation</a> section.
  * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `train.py` for options)

## Evaluation
To evaluate a trained network:

```Shell
python eval.py
```

You can specify the parameters listed in the `eval.py` file by flagging them or manually changing them.  


<img align="left" src= "https://github.com/amdegroot/ssd.pytorch/blob/master/doc/detection_examples.png">

## Performance

#### VOC2007 Test

##### mAP

| Original | Converted weiliu89 weights | From scratch w/o data aug | From scratch w/ data aug |
|:-:|:-:|:-:|:-:|
| 77.2 % | 77.26 % | 58.12% | 77.43 % |

##### FPS
**GTX 1060:** ~45.45 FPS

## Demos

### Use a pre-trained SSD network for detection

#### Download a pre-trained network
- We are trying to provide PyTorch `state_dicts` (dict of weight tensors) of the latest SSD model definitions trained on different datasets.  
- Currently, we provide the following PyTorch models:
    * SSD300 trained on VOC0712 (newest PyTorch weights)
      - https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth
    * SSD300 trained on VOC0712 (original Caffe weights)
      - https://s3.amazonaws.com/amdegroot-models/ssd_300_VOC0712.pth
- Our goal is to reproduce this table from the [original paper](http://arxiv.org/abs/1512.02325)
<p align="left">
<img src="http://www.cs.unc.edu/~wliu/papers/ssd_results.png" alt="SSD results on multiple datasets" width="800px"></p>

### Try the demo notebook
- Make sure you have [jupyter notebook](http://jupyter.readthedocs.io/en/latest/install.html) installed.
- Two alternatives for installing jupyter notebook:
    1. If you installed PyTorch with [conda](https://www.continuum.io/downloads) (recommended), then you should already have it.  (Just  navigate to the ssd.pytorch cloned repo and run):
    `jupyter notebook`

    2. If using [pip](https://pypi.python.org/pypi/pip):

```Shell
# make sure pip is upgraded
pip3 install --upgrade pip
# install jupyter notebook
pip install jupyter
# Run this inside ssd.pytorch
jupyter notebook
```

- Now navigate to `demo/demo.ipynb` at http://localhost:8888 (by default) and have at it!

### Try the webcam demo （经测试，存在 bug ）
- Works on CPU (may have to tweak `cv2.waitkey` for optimal fps) or on an NVIDIA GPU
- This demo currently requires opencv2+ w/ python bindings and an onboard webcam
  * You can change the default webcam in `demo/live.py`
- Install the [imutils](https://github.com/jrosebr1/imutils) package to leverage multi-threading on CPU:
  * `pip install imutils`
- Running `python -m demo.live` opens the webcam and begins detecting!

## TODO
We have accumulated the following to-do list, which we hope to complete in the near future
- Still to come:
  * [x] Support for the MS COCO dataset
  * [ ] Support for SSD512 training and testing
  * [ ] Support for training on custom datasets

## Authors

* [**Max deGroot**](https://github.com/amdegroot)
* [**Ellis Brown**](http://github.com/ellisbrown)

***Note:*** Unfortunately, this is just a hobby of ours and not a full-time job, so we'll do our best to keep things up to date, but no guarantees.  That being said, thanks to everyone for your continued help and feedback as it is really appreciated. We will try to address everything as soon as possible.

## References
- Wei Liu, et al. "SSD: Single Shot MultiBox Detector." [ECCV2016]((http://arxiv.org/abs/1512.02325)).
- [Original Implementation (CAFFE)](https://github.com/weiliu89/caffe/tree/ssd)
- A huge thank you to [Alex Koltun](https://github.com/alexkoltun) and his team at [Webyclip](webyclip.com) for their help in finishing the data augmentation portion.
- A list of other great SSD ports that were sources of inspiration (especially the Chainer repo):
  * [Chainer](https://github.com/Hakuyume/chainer-ssd), [Keras](https://github.com/rykov8/ssd_keras), [MXNet](https://github.com/zhreshold/mxnet-ssd), [Tensorflow](https://github.com/balancap/SSD-Tensorflow)


# 本人新增内容

 - 环境：

| python版本 | pytorch版本 |
|------------|-------------|
| 3.5        | 0.3.0       |

- 说明：
 
运行train.py之前请确保启动可视化工具visdom

## 总体思路

- 1、数据预处理
- 2、网络模型搭建
- 3、损失函数定义

#### 1、数据预处理

- 读取图像及对应xml,返回经过处理的一张图像及对应的真值框和类别

#### 2、网络结构搭建

- 总体结构

![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-6-13/79572279.jpg)

- 详细结构

![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-6-13/78236504.jpg)

- 各网络具体结构

vgg基础网络

![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-6-14/26832065.jpg)

extras新增层

![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-6-14/45744439.jpg)

head(loc定位、conf分类)

![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-6-14/90060469.jpg)

loc定位

![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-6-14/73834320.jpg)

conf分类

![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-6-14/18098016.jpg)


- 网络细节

当训练时，网络模型返回loc、conf、priors

一张图片（若干feature map）共生成8732个锚

loc： 通过网络输出的定位的预测 [32,8732,4] 

conf：  通过网络输出的分类的预测 [32,8732,21] 

priors：不同feature map根据公式生成的锚结果 [8732,4]
（称之为之所以称为锚，而不叫预测框。是因为锚是通过公式生成的，而不是通过网络预测输出的）


#### 3、损失函数定义

- 分类损失

使用多类别softmax loss

- 回归损失

使用 Smooth L1 loss

匹配策略：

1、通过使用IOU最大来匹配每一个 真值框 与 锚，这样就能保证每一个真值框 与 唯一的一个 锚 对应起来。

2、之后又将 锚 与 每一个 真值框 配对，只要两者之间的 IOU 大于一个阈值，这里本文的阈值为 0.5。

这样的结果是 每个真实框对应多个预测框。

![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-6-14/19492867.jpg)

Hard negative mining（硬性负开采）：

1、先将每一个物体位置上是 负样本 的 锚框 按照  confidence 的大小进行排序

2、选择最高的几个，保证最后 negatives、positives 的比例在 3:1。

这样的比例可以更快的优化，训练也更稳定。


## 结果

- 左侧为原版提供的[ssd300_mAP_77.43_v2.pth](https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth)

- 右侧为自己训练100000个batch结果[ssd300_VOC_100000_mAP_75.57](https://pan.baidu.com/s/1CC_QQBkiKRrkW6l6zpnFQQ)

<div align="center">
<img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-6-8/85403151.jpg" height="300px" alt="图片说明" ><img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-6-8/68841398.jpg" height="300px" alt="图片说明" > 
</div>

<div align="center">
<img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-6-8/20044012.jpg" height="300px" alt="图片说明" ><img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-6-14/14161052.jpg" height="300px" alt="图片说明" > 
</div>