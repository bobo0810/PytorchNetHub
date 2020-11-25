# 待解决
1. 一个节点用 多个GPU
https://pytorch.apachecn.org/docs/1.4/34.html

# 问题
1. batch_size和lr
    每个GPU的batch= 有效batch / 总GPUs
    lr无需修改
2. 如何验证和保存？
   > 验证代码无需修改，保证每个进程的log文件名称不同，最后只可视化rank=0  
    只保存rank=0的模型
3. 数据读取问题
    ```python
    TypeError: can't pickle Environment objects
    ```
   解决办法:DataLoader内num_workers=0
 
4. BN同步
    ```python
    # 仅支持DDP
    import torch
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ```