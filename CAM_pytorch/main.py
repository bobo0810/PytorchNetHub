import torch as t
from utils.config import opt
import models
import cv2
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader  #数据加载器
from data.MyDataSet import MyDataSet
from torch.autograd import Variable
from utils.visualize import Visualizer  #可视化visdom
from torchnet import meter
from functools import partial
from torch.nn import functional as F
def train():
    print("开始训练")
    # 定义一个网络模型对象
    # 通过config文件中模型名称来加载模型
    netWork = getattr(models, opt.model)()
    print('当前使用的模型为'+opt.model)

    # 定义可视化对象
    vis = Visualizer(opt.model)

    # 先将模型加载到内存中，即CPU中
    if opt.load_model_path:
        netWork.load_state_dict(t.load(opt.load_model_path, map_location=lambda storage, loc: storage))
    # 将模型转到GPU
    if opt.use_gpu:
        netWork.cuda()
        cudnn.benchmark = True

    train_data = MyDataSet(opt.dataset_root, train=True)
    val_data = MyDataSet(opt.dataset_root, train=False)
    # 数据集加载器
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=True,num_workers=opt.num_workers)

    # criterion 损失函数和optimizer优化器
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.SGD(netWork.parameters(), lr=lr, weight_decay=1e-6, momentum=0.5, nesterov=True)

    # 定义初始的loss
    previous_loss = float('inf')

    for epoch in range(opt.max_epoch):
        # 迭代数据集加载器
        for ii, (data_origin, label) in enumerate(train_dataloader):
            # input_img为模型输入图像
            input_img = Variable(data_origin)
            # label_img为对应标签
            label_img = Variable(label)
            # 将数据转到GPU
            if opt.use_gpu:
                input_img = input_img.cuda()
                label_img = label_img.cuda()
            # 优化器梯度清零
            optimizer.zero_grad()
            label_output = netWork(input_img)
            # 损失为交叉熵
            loss = criterion(label_output, label_img)
            loss.backward()
            optimizer.step()
            current_loss = loss.data[0]
            # 每8次可视化loss
            if ii % 8 == 0.0:
                vis.plot('loss', current_loss)

        #  一个epoch之后保存模型
        t.save(netWork.state_dict(), opt.checkpoint_root + opt.model + '_' + str(epoch) + '.pth')
        print("第" + str(epoch) + "次epoch完成==========================")
        vis.log("epoch:{epoch},lr:{lr},loss:{loss}".format(
            epoch=epoch, loss=current_loss, lr= lr ))

        # 更新学习率  如果训练集损失开始升高，则降低学习率
        if current_loss > previous_loss:
            lr = lr * opt.lr_decay
            # 不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = current_loss


        #============验证===================
        # 模型调整为验证模式
        netWork.eval()
        confusion_matrix = meter.ConfusionMeter(2)
        for ii, (val_data_origin, val_label) in enumerate(val_dataloader):
            val_input_img = Variable(val_data_origin, volatile=True)
            val_label_img = Variable(val_label.type(t.LongTensor), volatile=True)
            # 将数据转到GPU
            if opt.use_gpu:
                val_input_img = val_input_img.cuda()
            val_pridict_label = netWork(val_input_img)
            confusion_matrix.add(val_pridict_label.data.squeeze(), val_label.type(t.LongTensor))
        # cm_value   混淆矩阵的值
        cm_value = confusion_matrix.value()
        # 预测正确的数量除以总数量 再*100  得到正确率
        accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
        vis.plot('验证集准确率', accuracy)
        # 将模型调整为训练模式
        netWork.train()
    print("============训练完毕=============")

def visualize_cam():
    # 通过config文件中模型名称来加载模型
    netWork = getattr(models, opt.model)()
    print('当前使用的模型为' + opt.model)

    # 先将模型加载到内存中，即CPU中
    if opt.load_model_path:
        print('加载预训练模型')
        netWork.load_state_dict(t.load(opt.load_model_path, map_location=lambda storage, loc: storage))

    netWork.eval()
    # 为最后一个conv层添加一个钩子
    feature_blob = []
    netWork.feature_layer.register_forward_hook(partial(hook, feature_blob = feature_blob))

    # netWork._modules.get('feature_layer.29').register_forward_hook(partial(hook, feature_blob = feature_blob))

    # 得到 softmax(即 最后一层fc层)权重
    params = list(netWork.parameters())
    softmax_weight = np.squeeze(params[-2].data.numpy())

    #==============================================================
    # [3,64, 128]
    img_origin, img = MyDataSet(opt.dataset_root, train=True).get_test_img()



    #[1,2]
    output = netWork(Variable(img).unsqueeze(0))
    probs = F.softmax(output).data.squeeze()
    # 按照 概率进行从大到小排序
    probs, idx = probs.sort(0, descending = True)

    # generate class activation map for the top-5 prediction
    # 对于 top-5 的预测 生成CAM  类激活映射
    # feature_blob[0]  为最后一个conv层输出的结果  [1,2048,7,7]
    # softmax_weight  为全连接的权重 [1000,2048]
    # idx[0: 5]  为 按照概率从大到小排序后的 前五个概率的下标
    # cams list:5  每个里面都是[224,224]

    id_bobo = []
    id_bobo.append(idx[0])
    cams = compute_cam(feature_blob[0], softmax_weight,id_bobo )
    for i in range(len(cams)):
        # render cam and original image
        # 将 cam类激活映射 和 原图  组合在一起
        filename ='out.jpg'

        # 原图 [216,380,3]
        h, w, _ = img_origin.shape

        # 从cams类激活映射 转化的 热力图 [216,380,3]
        heatmap = cv2.applyColorMap(cv2.resize(cams[i], (w, h)), cv2.COLORMAP_JET)

        # 最终结果= 0.3热力图  +  0.5原图
        result = heatmap * 0.3 + img_origin * 0.5
        cv2.imwrite(filename, result)


# a hook to a given layer
def hook(module, input, output, feature_blob):
    feature_blob.append(output.data.numpy())

# compute class activation map
def compute_cam(activation, softmax_weight, class_ids):
    '''
     对于 top-5 的预测 生成CAM  类激活映射
    :param activation:  最后一个conv层输出的结果  [1,2048,7,7]
    :param softmax_weight: 全连接的权重 [1000,2048]
    :param class_ids: 按照概率从大到小排序后的 概率的下标
    '''
    # b:  batch size
    # c:  channel 通道数
    b, c, h, w = activation.shape
    cams = []
    for idx in class_ids:
        #[2048,49]
        activation = activation.reshape(c, h * w)
        # dot 计算两个张量的点乘 (内积). 矩阵乘法
        # [1.49]
        cam = softmax_weight[idx].dot(activation)
        # [7.7]
        cam = cam.reshape(h, w)
        # 归一化 normalize to [0, 1]
        cam =  (cam - cam.min()) / (cam.max() - cam.min())
        # 转化到0-255之间 conver to [0, 255]
        cam = np.uint8(255 * cam)
        # reshape to (224, 224)
        # 从[7,7] resize到[224,224]
        cams.append(cv2.resize(cam, (64, 128)))

    return cams

if __name__ == '__main__':
    #开始训练
    # train()

    # 可视化class_activation_map
    visualize_cam()
