# DistributedDataParallel

## 说明
- 分布式数据并行DDP最小实现
- 适用单机多卡、多机多卡训练

## 运行

### 示例
```PowerShell
//只有所有节点执行Shell命令，才开始训练
python ddp.py --nodes 节点数 --gpus 每个节点的GPU数量 --nr 当前节点序号  --ip 当前节点ip 
```

### 单机多卡 
节点ip=192.168.3.8
```PowerShell
Shell: CUDA_VISIBLE_DEVICES=0,1 python ddp.py --nodes 1 --gpus 2 --nr 0  --ip 192.168.3.8
```

### 多机多卡 
主节点ip=192.168.3.8  
```PowerShell
主节点Shell: CUDA_VISIBLE_DEVICES=0,1  python ddp.py --nodes 2 --gpus 2 --nr 0  --ip 192.168.3.8 
副节点Shell: CUDA_VISIBLE_DEVICES=0,1  python ddp.py --nodes 2 --gpus 2 --nr 1  --ip 192.168.3.8 
```


## 总结问题

1. batch_size

   > 有效batch = 每个GPU的batch * 总GPUs

2. 验证、保存

   > 验证：确保不同进程保存的log名称不同，最后只可视化rank=0。  
   > 保存：只保存rank=0的模型。
   
3. 数据读取

   - DataLoader采用Lmdb读取，若如下错误

     ```python
     TypeError: can't pickle Environment objects
     ```

     > 解决办法:DataLoader内num_workers=0

   - DataLoader采用其他方式读取，若如下错误

     ```python
     Attribute:Can’t pickle local object ‘DataLoader.__init__.<locals>.<lambda>’
     ```

     > 解决办法:   lambda x: Image.fromarray(x)   改为   Image.fromarray

4. 同步BN

    ```python
    # 仅支持DDP
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ```

# 参考
[Distributed data parallel training in Pytorch](https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html) 推荐！

[distributed_tutorial](https://github.com/yangkky/distributed_tutorial/blob/master/src/mnist-distributed.py)

[PyTorch分布式训练简明教程](https://zhuanlan.zhihu.com/p/113694038)

[Pytorch 分布式训练](https://zhuanlan.zhihu.com/p/76638962) 

[discuss.pytorch](https://discuss.pytorch.org/t/cant-pickle-local-object-dataloader-init-locals-lambda/31857) 