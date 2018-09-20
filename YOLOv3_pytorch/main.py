from __future__ import division
from torch.utils.data import DataLoader
import torch.optim as optim
from models.models import *
from utils.utils import *
from datasets.datasets import *
import os
import time
import datetime
import resource
from utils.visualize import Visualizer
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.config import opt_train,opt_test,opt_detect
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from torchnet import meter  #仪表  用来显示loss等图形
def train():
    opt = opt_train
    os.makedirs('checkpoints', exist_ok=True)  #用于存储训练后的网络模型

    # 定义可视化对象
    vis = Visualizer('YOLO v3 on coco')

    # Get data configuration
    # 获取dataloader配置
    data_config     = parse_data_config(opt.data_config_path)
    num_classes = int(data_config['classes'])
    # 拿到训练集
    train_path      = data_config['train']

    #hyperparams 即cfg中的[net]部分，网络训练的超参数
    hyperparams     = parse_model_config(opt.model_config_path)[0]   # model_config_path：模型网络结构cf文件
    learning_rate   = float(hyperparams['learning_rate'])
    momentum        = float(hyperparams['momentum'])
    decay           = float(hyperparams['decay'])
    burn_in         = int(hyperparams['burn_in'])

    # Initiate model
    # 初始化模型
    model = Darknet(opt.model_config_path)   # model_config_path：模型网络结构

   # 训练不再保存为.weights格式模型
    if opt.load_model_path:
        # 加载预训练好的的yolo v3模型  和 优化器状态
        print('加载已训练好的yolo v3模型')
        #多GPU
        checkpoint = torch.load(opt.load_model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        # 多GPU有问题，不可用
        # if torch.cuda.device_count() > 1:
        #     print('Using ', torch.cuda.device_count(), ' GPUs')
        #     model = nn.DataParallel(model)
        model.train() # 转为训练模式

        del checkpoint  # current, saved
    else:
        print('初始化yolo v3模型参数')
        model.apply(weights_init_normal)
        # 多GPU有问题，不可用
        # if torch.cuda.device_count() > 1:
        #     print('Using ', torch.cuda.device_count(), ' GPUs')
        #     model = nn.DataParallel(model)
        model.train() # 转为训练模式


    if opt.use_cuda:
        model = model.cuda()
        torch.backends.cudnn.benchmark = True

    # # 打印模型信息
    # modelinfo(model)

    # Get dataloader
    dataloader = torch.utils.data.DataLoader(
        ListDataset(train_path),
        batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
    # 设置好 默认新建的tensor类型
    Tensor = torch.cuda.FloatTensor if opt.use_cuda else torch.FloatTensor
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=0, weight_decay=decay)
    
    # 计算所有书的平均数和标准差，来统计一个epoch中损失的平均值
    loss_meter = meter.AverageValueMeter()
    previous_loss = float('inf')  # 表示正无穷
    # 开始训练
    for epoch in range(opt.epochs):
       
        # 清空仪表信息和混淆矩阵信息
        loss_meter.reset()
        # 每轮epoch
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            # imgs :处理后的图像tensor[16,3,416,416]        targets:坐标被归一化后的真值框filled_labels[16,50,5] 值在0-1之间
            imgs = Variable(imgs.type(Tensor))
            targets = Variable(targets.type(Tensor), requires_grad=False)
            # 优化器梯度清零
            optimizer.zero_grad()
            # 得到网络输出值，作为损失 (loss :多尺度预测的总loss之和)
            loss = model(imgs, targets)
            # 反向传播  自动求梯度
            loss.backward()
            # 更新优化器的可学习参数
            optimizer.step()
            loss_meter.add(loss.item())
            # 总loss为 loss_x、loss_y、loss_w、loss_h、loss_conf、loss_cls之和
            print('[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f]' %
                                        (epoch, opt.epochs, batch_i, len(dataloader),
                                        model.losses['x'], model.losses['y'], model.losses['w'],
                                        model.losses['h'], model.losses['conf'], model.losses['cls'],
                                        loss.item(), model.losses['recall']))
            # 每print_freq次可视化loss
            if batch_i % opt.print_freq == opt.print_freq - 1:
                # plot是自定义的方法
                vis.plot('batch loss', loss.item())
                vis.plot('batch loss_x', model.losses['x'])
                vis.plot('batch loss_y', model.losses['y'])
                vis.plot('batch loss_w', model.losses['w'])
                vis.plot('batch loss_h', model.losses['h'])
                vis.plot('batch loss_conf', model.losses['conf'])
                vis.plot('batch loss_cls', model.losses['cls'])
                vis.plot('batch recall', model.losses['recall'])
        # 每隔几个模型保存一次
        if epoch % opt.checkpoint_interval == 0:
            checkpoint = {'model': model.state_dict()}
            torch.save(checkpoint, opt.checkpoint_dir + '/'+str(epoch)+'yolov3.pt')
        print("第" + str(epoch) + "次epoch完成==========================")
        # 当前时刻的一些信息
        vis.log("epoch:{epoch},lr:{lr},loss:{loss}".format(
            epoch=epoch, loss=str(loss.item()), lr=learning_rate))
        
        #以下为bobo添加，原作者代码 没有更新学习率参数，故添加。（也可能是弄巧成拙）
        #更新学习率  如果损失开始升高，则降低学习率
        if loss_meter.value()[0]>previous_loss:
            learning_rate = learning_rate * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        previous_loss=loss_meter.value()[0]
def test():

    opt = opt_test
    # Get data configuration
    data_config = parse_data_config(opt.data_config_path)
    test_path = data_config['valid']
    num_classes = int(data_config['classes'])

    # Initiate model
    model = Darknet(opt.model_config_path)

    if opt.load_model_path:
        # 判断是训练模型.pt 还是.weights官方模型
        if '.pt' in opt.load_model_path:
            # 加载预训练好的的yolo v3模型
            print('加载已训练好的yolo v3自训练模型')
            checkpoint = torch.load(opt.load_model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
        if '.weights' in opt.load_model_path:
            model.load_weights(opt.load_model_path)
    else:
        # 初始化Conv、BatchNorm2d权重
        print('初始化yolo v3模型参数')
        model.apply(weights_init_normal)
        
    if opt.use_cuda:
        model = model.cuda()

    model.eval()

    # Get dataloader
    dataset = ListDataset(test_path)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    Tensor = torch.cuda.FloatTensor if opt.use_cuda else torch.FloatTensor

    n_gt = 0
    correct = 0

    print('Compute mAP...')

    outputs = []
    targets = None
    APs = []
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = targets.type(Tensor)

        with torch.no_grad():
            output = model(imgs)
            output = non_max_suppression(output, 80, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)

        # Compute average precision for each sample
        for sample_i in range(targets.size(0)):
            correct = []

            # Get labels for sample where width is not zero (dummies)
            annotations = targets[sample_i, targets[sample_i, :, 3] != 0]
            # Extract detections
            detections = output[sample_i]

            if detections is None:
                # If there are no detections but there are annotations mask as zero AP
                if annotations.size(0) != 0:
                    APs.append(0)
                print('有物体，但是预测为空，则AP=0')
                continue

            # Get detections sorted by decreasing confidence scores
            detections = detections[np.argsort(-detections[:, 4])]

            # If no annotations add number of detections as incorrect
            if annotations.size(0) == 0:
                correct.extend([0 for _ in range(len(detections))])
            else:
                # Extract target boxes as (x1, y1, x2, y2)
                target_boxes = torch.FloatTensor(annotations[:, 1:].shape)
                target_boxes[:, 0] = (annotations[:, 1] - annotations[:, 3] / 2)
                target_boxes[:, 1] = (annotations[:, 2] - annotations[:, 4] / 2)
                target_boxes[:, 2] = (annotations[:, 1] + annotations[:, 3] / 2)
                target_boxes[:, 3] = (annotations[:, 2] + annotations[:, 4] / 2)
                target_boxes *= opt.img_size

                detected = []
                for *pred_bbox, conf, obj_conf, obj_pred in detections:

                    pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
                    # Compute iou with target boxes
                    iou = bbox_iou(pred_bbox, target_boxes)
                    # Extract index of largest overlap
                    best_i = np.argmax(iou)
                    # If overlap exceeds threshold and classification is correct mark as correct
                    if iou[best_i] > opt.iou_thres and obj_pred == annotations[best_i, 0] and best_i not in detected:
                        correct.append(1)
                        detected.append(best_i)
                    else:
                        correct.append(0)

            # Extract true and false positives
            true_positives = np.array(correct)
            false_positives = 1 - true_positives

            # Compute cumulative false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            # Compute recall and precision at all ranks
            recall = true_positives / annotations.size(0) if annotations.size(0) else true_positives
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # Compute average precision
            AP = compute_ap(recall, precision)
            APs.append(AP)

            print("+ Sample [%d/%d] AP: %.4f (%.4f)" % (len(APs), len(dataset), AP, np.mean(APs)))

    print("Mean Average Precision: %.4f" % np.mean(APs))
def  detect():


    opt = opt_detect
    print(opt)

    # 创建output文件夹，以便保存生成的结果
    os.makedirs('output', exist_ok=True)

    # Set up model
    model = Darknet(opt.config_path, img_size=opt.img_size)

    if opt.load_model_path:
        # 判断是训练模型.pt 还是.weights官方模型
        if '.pt' in opt.load_model_path:
            # 加载预训练好的的yolo v3模型
            print('加载已训练好的yolo v3自训练模型')
            checkpoint = torch.load(opt.load_model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
        if '.weights' in opt.load_model_path:
            model.load_weights(opt.load_model_path)
    else:
        # 初始化Conv、BatchNorm2d权重
        print('初始化yolo v3模型参数')
        model.apply(weights_init_normal)

    # 将模型转到GPU上
    if opt.use_cuda:
        model.cuda()

    # 模型调整为验证模式
    model.eval()

    # 加载数据集
    dataloader = DataLoader(ImageFolder(opt.image_folder, img_size=opt.img_size),
                            batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    # 从文件中提取类标签
    classes = load_classes(opt.class_path)

    # 设置默认Tensor
    Tensor = torch.cuda.FloatTensor if opt.use_cuda else torch.FloatTensor

    imgs = []  # 保存文件路径
    img_detections = []  # 存储每个图像索引的网络输出检测结果

    print('\n开始进行 目标检测:')
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            # 得到网络输出的结果
            detections = model(input_imgs)
            # 进行NMS 非极大值抑制
            detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)

        # 记录时间
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

        # 保存图片和 网络输出的检测结果
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print('\nSaving images:')
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.switch_backend('agg')
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # The amount of padding that was added
        pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
        # Image height and width after padding is removed
        unpad_h = opt.img_size - pad_y
        unpad_w = opt.img_size - pad_x

        # Draw bounding boxes and labels of detections
        if detections is not None:
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                print('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

                # Rescale coordinates to original dimensions
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                         edgecolor=color,
                                         facecolor='none')
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                         bbox={'color': color, 'pad': 0})

        # Save generated image with detections
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig('output/%d.png' % (img_i), bbox_inches='tight', pad_inches=0.0)
        plt.close()






if __name__ == '__main__':
    '''
    增大 进程打开的最大文件数，内核默认为1024
    RLIMIT_NOFILE一个进程能打开的最大文件数，内核默认是1024
    rlimit的两个参数
    soft limit  指内核所能支持的资源上限 最大也只能达到1024
    hard limit 学校机房值为4096  在资源中只是作为soft limit的上限。当你设置hard limit后，你以后设置的soft limit只能小于hard limit
    '''
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    # torch.cuda.set_device(1)

    train()   # 开始训练
    # test()    # 测试，计算mAP
    # detect()    # 分割demo