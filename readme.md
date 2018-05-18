

# 注意事项：
- 仅供看注释，代码不完整，不能执行。
- 写代码建议参考原pytorch-book

# pytorch书：
- dogvscat只加注释，不能运行  仅用于理解
- chapter7-GAN生成动漫头像


# 图像去噪论文复现：
- 论文地址
- http://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CJFQ&dbname=CJFDLAST2017&filename=JSJC201703042&uid=WEEvREcwSlJHSldRa1FhdXNXa0hIb3VVSnliNDU0a2dObEJYUVM1MzR2cz0=$9A4hF_YAuvQ5obgVAqNKPCYcEjKensW4ggI8Fm4gTkoUKaID8j8gFw!!&v=MTUzMzkxRnJDVVJMS2ZZdWRvRnk3blVydkJMejdCYmJHNEg5Yk1ySTlCWm9SOGVYMUx1eFlTN0RoMVQzcVRyV00=
- 验证集可视化有bug
- 效果


header 1 | header 2
---|---
第0次生成大的去噪图像 | 第20次网络生成的去噪图像
第50次网络生成的去噪图像 | 第99次网络生成的去噪图像



<figure class="half">
    <img src="https://note.youdao.com/yws/public/resource/16bce8e15505fe1dd7d5bf727617ca98/xmlnote/626A4A3D6A3E48AA88B6D952ED1716CB/3434">
    <img src="https://note.youdao.com/yws/public/resource/16bce8e15505fe1dd7d5bf727617ca98/xmlnote/80F82927EB434432B444B360D7E3A92A/3432">
</figure>
<figure class="half">
    <img src="https://note.youdao.com/yws/public/resource/16bce8e15505fe1dd7d5bf727617ca98/xmlnote/410D3174AF32459090916866DD4EFDD5/3433">
    <img src="https://note.youdao.com/yws/public/resource/16bce8e15505fe1dd7d5bf727617ca98/xmlnote/415AF810FA7143A9BC42DC21B9615AB0/3435">
</figure>




**loss**
![loss](https://note.youdao.com/yws/public/resource/16bce8e15505fe1dd7d5bf727617ca98/xmlnote/AA500408C7FC468A8AFF9F8D8CE78FB5/3456)


**参数**

![loss](https://note.youdao.com/yws/public/resource/16bce8e15505fe1dd7d5bf727617ca98/xmlnote/088B72F912084D6BA9D28F505D5D289F/3461)





