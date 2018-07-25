from SSD_pytorch.data import *
from SSD_pytorch.utils.augmentations import SSDAugmentation
from SSD_pytorch.models.modules import MultiBoxLoss
from SSD_pytorch.models.ssd import build_ssd
from SSD_pytorch.models.modules.init_weights import weights_init
from SSD_pytorch.data import VOC_CLASSES as VOC_CLASSES
import torch
from SSD_pytorch.utils.config import opt
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from SSD_pytorch.utils.visualize import Visualizer
from SSD_pytorch.utils.timer import Timer
from SSD_pytorch.utils.eval_untils import evaluate_detections
import os
import time
import sys
import pickle
from matplotlib import pyplot as plt
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET



#设置创建tensor的默认类型
if opt.use_gpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def train():

    #返回一个batch的信息，其中每行代表一张图像 及 对应的真值框和类别
    dataset = VOCDetection(root=opt.voc_data_root,
                           transform=SSDAugmentation(opt.voc['min_dim'],
                                                     opt.MEANS))
    # 定义可视化对象
    vis = Visualizer(opt.env + opt.model)
    # 初始化ssd模型
    #  opt.voc['min_dim']：300×300图像
    # opt.voc['num_classes']  21 类别数（20类+1背景）
    ssd_net = build_ssd('train', opt.voc['min_dim'], opt.voc['num_classes'])
    # 数据并行是当我们将小批量样品分成多个较小的批量批次，并且对每个较小的小批量并行运行计算。
    if opt.use_gpu:
        # GPU并行
        net = torch.nn.DataParallel(ssd_net)
        net = net.cuda()
        cudnn.benchmark = True
    # 加载预训练好的的SSD模型
    if opt.load_model_path:
        print('加载已训练好的SSD模型')
        ssd_net.load_state_dict(torch.load(opt.load_model_path,map_location=lambda storage, loc: storage))
        print('加载权重完成!')
    else:
        # 从头开始训练
        # 先加载去掉全连接层的vgg网络。权重为预训练好的 去掉全连接层的vgg网络权重
        vgg_weights = torch.load(opt.basenet)
        print('正在加载vgg基础网络...')
        ssd_net.vgg.load_state_dict(vgg_weights)

        print('从头开始训练，初始化除vgg之外的网络层权重...')
        # 使用xavier方法来初始化vgg后面的新增层、loc用于回归层、conf用于分类层  的权重
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    # 使用随机梯度下降SGD
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    # SSD的损失函数
    criterion = MultiBoxLoss(opt.voc['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, opt.use_gpu)
    # 将网络置为训练模式
    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('加载数据集...')
    # 一个epoch需要几次batch  //除法操作（向下取整）
    epoch_size = len(dataset) // opt.batch_size


    step_index = 0
    # 设置可视化
    if opt.visdom:
        # 标题
        vis_title = 'SSD.PyTorch on ' + dataset.name
        # 说明
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        # 每次迭代的图形
        iter_plot = vis.create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        # 每个epoch的图形
        epoch_plot = vis.create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)
    # 数据加载器
    # collate_fn合并样本列表以形成一个 mini-batch
    # pin_memory如果为True, 数据加载器会将张量复制到CUDA固定内存中, 然后再返回它们
    data_loader = data.DataLoader(dataset, opt.batch_size,
                                  num_workers=opt.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # 开始迭代数据集
    batch_iterator = iter(data_loader)
    # iteration数为每个batch迭代数
    #start_iter：从第几个iter开始迭代   max_iter：最大迭代数
    for iteration in range(opt.start_iter, opt.voc['max_iter']):
        # 可视化
        # epoch_size：一个epoch需要的batch数。即每一个epoch可视化一下
        if opt.visdom and iteration != 0 and (iteration % epoch_size == 0):
            vis.update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, 'append',
                            'append', epoch_size)
            # 重置epoch损失计数器
            loc_loss = 0
            conf_loss = 0
            epoch += 1
        # 如果迭代的batch数到达lr_steps的值时，则调整学习率
        if iteration in opt.voc['lr_steps']:
            step_index += 1
            lr = opt.lr * (opt.gamma ** (step_index))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # 加载数据集
        try:
            # batch_iterator 返回一个batch的信息，其中每行代表一张图像 及 对应的真值框和类别
            # target： [32,num_objs,5]   32：batch大小， num_objs ：一张图片的物体数，5：前四个数为坐标，最后一个数为类别
            images, targets = next(batch_iterator)
        # 遇到StopIteration，即迭代完一次数据集，需要再重头开始迭代
        except StopIteration:
            data_loader = data.DataLoader(dataset, opt.batch_size,
                                          num_workers=opt.num_workers,
                                          shuffle=True, collate_fn=detection_collate,
                                          pin_memory=True)
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)
        # 将训练集转到GPU上
        if opt.use_gpu:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward前向传播开始
        # t0、t1用于计算方法执行时间
        t0 = time.time()
        out = net(images)

        # 优化器梯度清零
        optimizer.zero_grad()
        # 损失函数得到定位误差和分类误差
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        # 反向传播
        loss.backward()
        # 更新可学习参数
        optimizer.step()
        t1 = time.time()

        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]
        # 每10个batch迭代输出信息
        if iteration % 10 == 0:
            print('前向+反向总共所需时间timer: %.4f sec.' % (t1 - t0))
            print('iter为 ' + repr(iteration) + ' || 总Loss: %.4f ||' % (loss.data[0]), end=' ')
        # 可视化
        if opt.visdom:
            vis.update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
                            iter_plot, epoch_plot, 'append')
        # 每5000次迭代保存模型
        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            ssd_net.saveSSD(str(iteration))

    # 保存最后的模型
    ssd_net.saveSSD()

def eval():
    '''
    使用验证集进行验证VOC2007测试集，计算AP及mAP
    '''

    # 加载网络
    num_classes = len(VOC_CLASSES) + 1  # +1 为背景
    net = build_ssd('test', 300, num_classes)  # 初始化SSD
    # 加载预训练好的的SSD模型
    if opt.load_model_path:
        print('加载已训练好的SSD模型')
        net.load_state_dict(torch.load(opt.load_model_path, map_location=lambda storage, loc: storage))
        print('加载权重完成!')
    #模型转为验证模式
    net.eval()
    # 加载数据 (使用VOC2007的测试集进行验证)
    dataset = VOCDetection(opt.voc_data_root, [('2007',  'test')],
                           BaseTransform(300, opt.MEANS),
                           VOCAnnotationTransform())
    if opt.use_gpu:
        net = net.cuda()
        cudnn.benchmark = True
    # 开始验证
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(VOC_CLASSES) + 1)]

    # 计算预测时间
    _t = {'im_detect': Timer(), 'misc': Timer()}
    # 保存预测结果的临时文件
    det_file = os.path.join(opt.temp, 'detections.pkl')

    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0))
        if opt.use_gpu:
            x = x.cuda()
        _t['im_detect'].tic()
        #得到预测结果
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        # 跳过j=0，因为它是背景
        # 遍历验证集 ，得到网络预测的结果
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.dim() == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets

        print('验证集图像检测进度: {:d}/{:d}  单张预测耗时：{:.3f}s'.format(i + 1,
                                                    num_images, detect_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('开始评估检测结果')
    evaluate_detections(all_boxes, opt.temp, dataset)


def test():
    '''
    测试集使用VOC2007测试集，将结果存储到voc2007_test.txt中
    '''
    num_classes = len(VOC_CLASSES) + 1  # +1 是因为背景
    net = build_ssd('test', 300, num_classes)  # 初始化SSD
    # 加载预训练好的的SSD模型
    if opt.load_model_path:
        print('加载已训练好的SSD模型')
        net.load_state_dict(torch.load(opt.load_model_path, map_location=lambda storage, loc: storage))
        print('加载权重完成!')

    # 模型调整为验证集
    net.eval()
    # 加载数据集
    testset = VOCDetection(opt.voc_data_root, [('2007', 'test')], None, VOCAnnotationTransform())
    if opt.use_gpu:
        net = net.cuda()
        cudnn.benchmark = True

    # dump predictions and assoc. ground truth to text file for now
    transform=BaseTransform(net.size, (104, 117, 123))
    filename = opt.temp_test + 'voc2007_test.txt'
    num_images = len(testset)
    for i in range(num_images):
        print('测试进度： {:d}/{:d}'.format(i + 1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        with open(filename, mode='a') as f:
            f.write('\nGROUND TRUTH FOR: ' + img_id + '\n')
            for box in annotation:
                f.write('label: ' + ' || '.join(str(b) for b in box) + '\n')
        if opt.use_gpu:
            x = x.cuda()

        y = net(x)  # forward pass
        detections = y.data
        # scale each detection back up to the image
        # 将每个检测缩放回图像
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS: ' + '\n')
                score = detections[0, i, j, 0]
                label_name = VOC_CLASSES[i - 1]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(pred_num) + ' label: ' + label_name + ' score: ' +
                            str(score) + ' ' + ' || '.join(str(c) for c in coords) + '\n')
                j += 1

def predict():
    '''
    使用SSD进行对象检测,并进行可视化
    '''
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    net = build_ssd('test', 300, 21)  # 初始化 SSD300，类别为21（20类别+1背景）
    if opt.load_model_path:
        print('加载已训练好的SSD模型')
        net.load_state_dict(torch.load(opt.load_model_path, map_location=lambda storage, loc: storage))
        print('加载权重完成!')

    # 加载原图像并可视化
    image = cv2.imread(opt.test_img, cv2.IMREAD_COLOR)
    #rgb格式的原图
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 显示原图
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)
    plt.title("origin img")
    plt.show()

    #对输入图像进行预处理
    #将图像resize到300x300
    x = cv2.resize(image, (300, 300)).astype(np.float32)
    #(104.0, 117.0, 123.0)为ImageNet数据集的三通道均值，更符合大自然的分布规律
    x -= (104.0, 117.0, 123.0)
    #类型转化
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    #显示预处理之后的图片
    plt.title("pre-process img")
    plt.imshow(x)
    plt.show()
    #  permute 按制定的维数排序
    x = torch.from_numpy(x).permute(2, 0, 1)

    #前向传播，得到网络预测
    xx = Variable(x.unsqueeze(0))  # 扩展第0维，因为网络的输入要求是一个batch
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)

    #解析网络输出并将结果可视化
    plt.figure(figsize=(10, 10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(rgb_image)  # 画出rgb格式的原图
    currentAxis = plt.gca()  #当前轴

    detections = y.data
    # 将检测结果缩放匹配到原图上  repeat：沿着指定的尺寸重复 tensor
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    for i in range(detections.size(1)):
        j = 0
        #置信度分数阈值设置为0.6
        while detections[0, i, j, 0] >= 0.6:
            score = detections[0, i, j, 0]
            label_name = VOC_CLASSES[i - 1]
            display_txt = '%s: %.2f' % (label_name, score)
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
            color = colors[i]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
            j += 1
    #可视化最后结果
    plt.title("predict img")
    plt.show()



if __name__ == '__main__':
    # train()  #使用VOC2007和2012的训练集+验证集 开始训练
    # eval()  # VOC2007测试集,计算各类AP及mAP
    # test()  #VOC2007测试集，将预测结果写入txt
    predict()  #可视化一张预测图片