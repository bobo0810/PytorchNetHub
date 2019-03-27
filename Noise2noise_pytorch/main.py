import os
from utils.config import opt_train,opt_test
from data.datasets import load_dataset
from noise2noise import Noise2Noise
def train():

    # 加载训练集和验证集
    train_loader = load_dataset(opt_train.train_dir, opt_train.train_size, opt_train, shuffled=True)
    valid_loader = load_dataset(opt_train.valid_dir, opt_train.valid_size, opt_train, shuffled=False)

    # 初始化模型并训练
    n2n = Noise2Noise(opt_train, trainable=True)
    n2n.train(train_loader, valid_loader)

def test():
    # 初始化模型，进行测试
    n2n = Noise2Noise(opt_test, trainable=False)
    opt_test.redux = False
    test_loader = load_dataset(opt_test.data,3, opt_test, shuffled=False, single=True)  #修改0
    n2n.load_model(opt_test.load_ckpt) #加载预训练模型
    n2n.test(test_loader)



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 选择哪块GPU运行 '0' or '1' or '0,1'

    #训练
    # train()

    # 测试单张图片，将结果保存到文件夹下
    test()
