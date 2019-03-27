# -*- coding:utf-8 -*-
# power by Mr.Li
# 设置训练时的默认参数
class Config_Train():


    # 数据集参数
    train_dir = '/home/bobo/data/noise2noise/train'  # 训练集地址
    valid_dir = '/home/bobo/data/noise2noise/valid'  # 验证集地址
    ckpt_save_path = './checkpoints'  # 模型保存地址
    report_interval = 250  # 每几次进行输出训练信息
    train_size = 1000  # 训练集大小
    valid_size = 200  # 验证集大小
    ckpt_overwrite=False

    # 超参数
    learning_rate = 0.001  # 初始学习率
    adam = [0.9, 0.99, 1e-8]  # 优化器
    batch_size = 4  # batch-size大小
    nb_epochs = 100  # 训练轮数
    loss = 'l1'  # 损失函数  选L1，倾向于学习中位数  可选'l1', 'l2'
    cuda = True  # 使用GPU

    # 生成噪声的参数
    noise_type = 'text'  # 选择噪声类型，选水印  可选'gaussian', 'poisson', 'text'
    noise_param = 0.5  # text噪声，即水印噪声选0.5，高斯\泊松噪声选50，
    crop_size = 128  # 对原图进行随机裁剪的尺寸，用于输入网络
    clean_targets = False  # 是否使用 干净目标作为标签 进行训练
    seed=False # 固定随机种子

# 设置测试时的默认参数
class Config_Test():


    # 数据集参数
    data= '/home/bobo/data/noise2noise/valid'
    load_ckpt='/home/bobo/windowsPycharmProject/noise2noise-pytorch/checkpoints/text-1142/n2n-epoch49-0.10109.pt'  #加载模型的地址
    cuda=True # 使用GPU



    # 生成噪声的参数
    noise_type = 'text'  # 选择噪声类型，选水印  可选'gaussian', 'poisson', 'text'
    noise_param = 0.5  # text噪声，即水印噪声选0.5, 高斯\泊松噪声选50，
    seed = False  # 固定随机种子
    crop_size = 256  # 对原图进行随机裁剪的尺寸，用于输入网络
    clean_targets=True # 测试时将 使用干净的标签真实 进行对比


# 初始化该类的一个对象
opt_train = Config_Train()
opt_test = Config_Test()