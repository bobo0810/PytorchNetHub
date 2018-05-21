# 代码注释

标签（空格分隔）： 以便理解

---

# 注意事项：
- 仅供看注释，代码不完整，不能执行。
- 写代码建议参考原pytorch-book


----------


# pytorch书：
- dogvscat只加注释，不能运行  仅用于理解
- chapter7-GAN生成动漫头像


----------

# YOLO V1版本：
- [引用地址][1]
- 目前训练可用，即train.py

----------


# 图像去噪论文复现：

 - 实现：papersReproduced

 - 论文地址
 [基于深度卷积神经网络的图像去噪研究][2]

 - 验证集可视化有bug
 - 效果


header 1 | header 2
---|---
第0次生成大的去噪图像 | 第20次网络生成的去噪图像
第50次网络生成的去噪图像 | 第99次网络生成的去噪图像


<div align="center">
<img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-18/35562877.jpg" height="300px" alt="图片说明" ><img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-18/35562877.jpg" height="300px" alt="图片说明" > 
<img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-18/94087410.jpg" height="300px" alt="图片说明" ><img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-18/20206610.jpg" height="300px" alt="图片说明" > 

</div>

**loss**

![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-18/9788775.jpg)


**参数**

![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-18/41620436.jpg)


  
  


  [1]: https://github.com/xiongzihua/pytorch-YOLO-v1
  [2]: http://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CJFQ&dbname=CJFDLAST2017&filename=JSJC201703042&uid=WEEvREcwSlJHSldRa1FhdXNXa0hIb3VVSnliNDU0a2dObEJYUVM1MzR2cz0=$9A4hF_YAuvQ5obgVAqNKPCYcEjKensW4ggI8Fm4gTkoUKaID8j8gFw!!&v=MTUzMzkxRnJDVVJMS2ZZdWRvRnk3blVydkJMejdCYmJHNEg5Yk1ySTlCWm9SOGVYMUx1eFlTN0RoMVQzcVRyV00=