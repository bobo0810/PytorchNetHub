from __future__ import division
from torch.utils.data import DataLoader
import torch.optim as optim
from models.models import *
from utils.utils import *
from datasets.datasets import *
import os
import time
import datetime
from utils.visualize import Visualizer
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.config import opt_train,opt_test,opt_detect
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
def train():

    opt = opt_train
    cuda = torch.cuda.is_available() and opt.use_cuda
    os.makedirs('output', exist_ok=True)  # 用于运行分割denmo时保存结果
    os.makedirs('checkpoints', exist_ok=True)  #用于存储训练后的网络模型

    classes = load_classes(opt.class_path) #coco数据集类别标签

    # Get data configuration
    # 获取dataloader配置
    data_config     = parse_data_config(opt.data_config_path)
    # 拿到训练集
    train_path      = data_config['train']

    # Get hyper parameters

    #hyperparams 即cfg中的[net]部分，网络训练的超参数
    hyperparams     = parse_model_config(opt.model_config_path)[0]   # model_config_path：模型网络结构cf文件
    learning_rate   = float(hyperparams['learning_rate'])
    momentum        = float(hyperparams['momentum'])
    decay           = float(hyperparams['decay'])
    burn_in         = int(hyperparams['burn_in'])

    # Initiate model
    # 初始化模型
    model = Darknet(opt.model_config_path) # model_config_path：模型网络结构
    if opt.load_model_path:
        # 加载预训练好的的yolo v3模型
        print('加载已训练好的yolo v3模型')
        model.load_weights(opt.load_model_path)
    else:
        # 初始化Conv、BatchNorm2d权重
        print('初始化yolo v3模型参数')
        model.apply(weights_init_normal)

    if cuda:
        model = model.cuda()
    # 可视化visdom
    if opt.visdom:
        # 定义可视化对象
        vis = Visualizer()
        # 标题
        vis_title = 'YOLO v3.PyTorch on '+opt.data_config_path
        # 说明
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        # 每次迭代的图形
        iter_plot = vis.create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)



    # 将模型调整为训练模式
    model.train()

    # Get dataloader
    dataloader = torch.utils.data.DataLoader(
        ListDataset(train_path),
        batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)


    # 设置好 默认新建的tensor类型
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=0, weight_decay=decay)

    # 开始训练
    for epoch in range(opt.epochs):
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

            # 总loss为 loss_x、loss_y、loss_w、loss_h、loss_conf、loss_cls之和
            print('[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f]' %
                                        (epoch, opt.epochs, batch_i, len(dataloader),
                                        model.losses['x'], model.losses['y'], model.losses['w'],
                                        model.losses['h'], model.losses['conf'], model.losses['cls'],
                                        loss.item(), model.losses['recall']))

            # 可视化
            if opt.visdom:
                vis.update_vis_plot(batch_i*(epoch+1),  model.losses['cls'], model.losses['conf'],loss.item(),iter_plot, 'append')

            # 统计 训练过程共使用多少张图片，用于 保存权重时写入 头文件中
            model.seen += imgs.size(0)
        # 每隔几个模型保存一次
        if epoch % opt.checkpoint_interval == 0:
            model.save_weights('%s/%d.weights' % (opt.checkpoint_dir, epoch))

def test():

    opt = opt_test
    cuda = torch.cuda.is_available() and opt.use_cuda
    # Get data configuration
    data_config = parse_data_config(opt.data_config_path)
    test_path = data_config['valid']
    num_classes = int(data_config['classes'])

    # Initiate model
    model = Darknet(opt.model_config_path)

    if opt.load_model_path:
        # 加载预训练好的的yolo v3模型
        print('加载已训练好的yolo v3模型')
        model.load_weights(opt.load_model_path)
    else:
        # 初始化Conv、BatchNorm2d权重
        print('初始化yolo v3模型参数')
        model.apply(weights_init_normal)

    if cuda:
        model = model.cuda()

    model.eval()

    # Get dataloader
    dataset = ListDataset(test_path)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

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

    cuda = torch.cuda.is_available() and opt.use_cuda
    # 创建output文件夹，以便保存生成的结果
    os.makedirs('output', exist_ok=True)

    # Set up model
    model = Darknet(opt.config_path, img_size=opt.img_size)

    if opt.load_model_path:
        # 加载预训练好的的yolo v3模型
        print('加载已训练好的yolo v3模型')
        model.load_weights(opt.load_model_path)
    else:
        # 初始化Conv、BatchNorm2d权重
        print('初始化yolo v3模型参数')
        model.apply(weights_init_normal)


    # 将模型转到GPU上
    if cuda:
        model.cuda()

    # 模型调整为验证模式
    model.eval()

    # 加载数据集
    dataloader = DataLoader(ImageFolder(opt.image_folder, img_size=opt.img_size),
                            batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    # 从文件中提取类标签
    classes = load_classes(opt.class_path)

    # 设置默认Tensor
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

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
    # train()   # 开始训练
    # test()    # 测试，计算mAP
    detect()    # 分割demo