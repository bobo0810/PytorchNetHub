# VisCNN_CVPR_2016_Loc
A PyTorch implementation of paper "Learning Deep Features for Discriminative Localization"

## Brief Introduction
```
从 https://github.com/metalbubble/CAM 中节选的内容，仅用于理解可视化CNN定位
```

## Paper
```
@article{zhou2015cnnlocalization,
  title={{Learning Deep Features for Discriminative Localization.}},
  author={Zhou, B. and Khosla, A. and Lapedriza. A. and Oliva, A. and Torralba, A.},
  journal={CVPR},
  year={2016}
}
```
# 结果
![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-10-8/76093830.jpg)

# 测试图
![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-10-8/93370027.jpg)

# 按照top-5概率的热力图

### top-1  类别预测为mountain bike, all-terrain bike, off-roader
![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-10-8/94679763.jpg)

###  top-2  类别预测为bicycle-built-for-two, tandem bicycle, tandem
![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-10-8/71113337.jpg)

###  top-3  类别预测为unicycle, monocycle
![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-10-8/77864050.jpg)

###  top-4  类别预测为seashore, coast, seacoast, sea-coast

![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-10-8/81165493.jpg)

###  top-5  类别预测为alp
![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-10-8/57134571.jpg)