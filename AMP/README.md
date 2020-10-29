# AMP: Automatic Mixed Precision

## 说明
- 好处：多快好省, batch增大
- 训练：DataParallel且梯度累加 的代码

## 注意
- AMP保存的模型仍为FP32
- AMP下模型保存两份权重。

    FP16权重用于反向传播计算（加速训练），并更新参数在FP32权重上（主模型）
- 若想推理加速，在精度接受范围内img\model手动half()为FP16，然后只能GPU推理

## 环境

| python版本 | pytorch版本 | 系统   |
|------------|-------------|--------|
| 3.6        | >=1.6.0       | Ubuntu |


## 参考
[pytorch_docs](https://pytorch.org/docs/stable/notes/amp_examples.html)
[基于Apex的混合精度加速](https://zhuanlan.zhihu.com/p/79887894)
[论文精读：Mixed Precision Training](https://zhuanlan.zhihu.com/p/163493798)
