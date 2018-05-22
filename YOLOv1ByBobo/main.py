from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import opt
from data.dataset import yoloDataset
from models.net import vgg16
from utils.visualize import Visualizer
from utils.yoloLoss import yoloLoss


def train():
    vis=Visualizer(opt.env)
    # 网络部分======================================================开始
    # True则返回预训练好的VGG16模型
    net = vgg16(pretrained=True)
    # 修改vgg16的分类部分
    net.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        # nn.Linear(4096, 4096),
        # nn.ReLU(True),
        # nn.Dropout(),
        nn.Linear(4096, 1470),
    )
    # 初始化网络的线性层 权重及偏向
    for m in net.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
    # 将模型转移到GPU上
    if opt.use_gpu:
        net.cuda()
    # 输出网络结构    debug时查看时否相同
    print(net)
    print('加载好预先训练好的模型')
    # 将模型调整为训练模式
    net.train()
    # 网络部分======================================================结束

    #加载数据部分====================================================开始
    # 自定义封装数据集
    train_dataset = yoloDataset(root=opt.file_root, list_file=opt.voc_2012train, train=True, transform=[transforms.ToTensor()])
    # 数据集加载器  shuffle：打乱顺序    num_workers：线程数
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    test_dataset = yoloDataset(root=opt.test_root, list_file=opt.voc_2007test, train=False, transform=[transforms.ToTensor()])
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
    # 加载数据部分====================================================结束

    # 交叉熵
    criterion = yoloLoss(7, 2, 5, 0.5)
    # 优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)

    print('训练集有 %d 张图像' % (len(train_dataset)))
    print('the batch_size is %d' % (opt.batch_size))
    # 将训练过程的信息写入log文件中
    logfile = open('log.txt', 'w')
    best_test_loss = np.inf

    for epoch in range(opt.num_epochs):
        if epoch == 1:
            opt.learning_rate = 0.0005
        if epoch == 2:
            opt.learning_rate = 0.00075
        if epoch == 3:
            opt.learning_rate = 0.001
        if epoch == 80:
            opt.learning_rate = 0.0001
        if epoch == 100:
            opt.learning_rate = 0.00001
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.learning_rate
        # 第几次epoch  及 当前epoch的学习率
        print('\n\n当前的epoch为 %d / %d' % (epoch + 1, opt.num_epochs))
        print('当前epoch的学习率: {}'.format(opt.learning_rate))

        # 每轮epoch的总loss
        total_loss = 0.
        # 开始训练
        for i, (images, target) in enumerate(train_loader):
            images = Variable(images)
            target = Variable(target)
            if opt.use_gpu:
                images, target = images.cuda(), target.cuda()
            # 前向传播，得到预测值
            pred = net(images)
            # 计算损失
            loss = criterion(pred, target)
            total_loss += loss.data[0]
            # 优化器梯度清零
            optimizer.zero_grad()
            #loss反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            if (i + 1) % opt.print_freq == 0:
                print('当前epoch为 [%d/%d], Iter [%d/%d] 当前batch损失为: %.4f, 当前epoch到目前为止平均损失为: %.4f'
                      % (epoch + 1, opt.num_epochs, i + 1, len(train_loader), loss.data[0], total_loss / (i + 1)))
                # 画出训练集的平均损失
                vis.plot_train_val(loss_train=total_loss / (i + 1))
# =========================================================看到此
        # 一次epoch验证
        validation_loss = 0.0
        # 模型调整为验证模式
        net.eval()
        # 每轮epoch之后用VOC2007测试集进行验证
        for i, (images, target) in enumerate(test_loader):
            images = Variable(images, volatile=True)
            target = Variable(target, volatile=True)
            if opt.use_gpu:
                images, target = images.cuda(), target.cuda()
            # 前向传播得到预测值
            pred = net(images)
            # loss
            loss = criterion(pred, target)
            validation_loss += loss.data[0]
        # 计算在VOC2007测试集上的平均损失
        validation_loss /= len(test_loader)
        # 画出验证集的平均损失
        vis.plot_train_val(loss_val=validation_loss)
        # 训练模型的目标是 在验证集上的loss最小
        # 保存到目前为止 在验证集上的loss最小 的模型
        if best_test_loss > validation_loss:
            best_test_loss = validation_loss
            print('得到最好的验证集的平均损失  %.5f' % best_test_loss)
            torch.save(net.state_dict(),opt.best_test_loss_model_path)
        # 将当前epoch的参数写入log文件中
        logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')
        logfile.flush()
        # 保存最新的模型
        torch.save(net.state_dict(),opt.current_epoch_model_path)



def predict():

    # fasle 返回 未训练好的模型
    predict_model = vgg16(pretrained=False)
    # 修改网络结构
    predict_model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                #nn.Linear(4096, 4096),
                #nn.ReLU(True),
                #nn.Dropout(),
                nn.Linear(4096, 1470),
            )
    predict_model.load_state_dict(torch.load(opt.load_model_path))
    # 模型改为预测模式
    predict_model.eval()
    if opt.use_gpu:
        predict_model.cuda()
    # 测试集照片地址
    test_img_dir=opt.test_img_dir
    image = cv2.imread(test_img_dir)
    result = predict_gpu(predict_model, test_img_dir)
    for left_up, right_bottom, class_name, _, prob in result:
        # 将预测框添加到测试图片中
        cv2.rectangle(image, left_up, right_bottom, (0, 255, 0), 2)
        cv2.putText(image, class_name, left_up, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        print(prob)
    # 将测试结果写入
    cv2.imwrite(opt.result_img_dir,image)


def predict_gpu(model,image_name,root_path=''):
    '''
    预测一张测试照片
    '''
    result = []
    aaaa=root_path+image_name
    image = cv2.imread(root_path+image_name)
    h,w,_ = image.shape
    img = cv2.resize(image,(224,224))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    mean = (123,117,104)#RGB
    img = img - np.array(mean,dtype=np.float32)

    transform = transforms.Compose([transforms.ToTensor(),])
    img = transform(img)
    img = Variable(img[None,:,:,:],volatile=True)
    img = img.cuda()

    pred = model(img) #1x7x7x30
    pred = pred.cpu()
    boxes,cls_indexs,probs =  decoder(pred)

    for i,box in enumerate(boxes):
        x1 = int(box[0]*w)
        x2 = int(box[2]*w)
        y1 = int(box[1]*h)
        y2 = int(box[3]*h)
        cls_index = cls_indexs[i]
        cls_index = int(cls_index) # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(x1,y1),(x2,y2),opt.VOC_CLASSES[cls_index],image_name,prob])
    return result

def decoder(pred):
    '''
    pred (tensor) 1x7x7x30
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''
    boxes=[]
    cls_indexs=[]
    probs = []
    cell_size = 1./7
    pred = pred.data
    pred = pred.squeeze(0) #7x7x30
    contain1 = pred[:,:,4].unsqueeze(2)
    contain2 = pred[:,:,9].unsqueeze(2)
    contain = torch.cat((contain1,contain2),2)
    mask1 = contain > 0.9 #大于阈值
    mask2 = (contain==contain.max()) #we always select the best contain_prob what ever it>0.9
    mask = (mask1+mask2).gt(0)
    min_score,min_index = torch.min(mask,2) #每个cell只选最大概率的那个预测框
    for i in range(7):
        for j in range(7):
            for b in range(2):
                index = min_index[i,j]
                mask[i,j,index] = 0
                if mask[i,j,b] == 1:
                    #print(i,j,b)
                    box = pred[i,j,b*5:b*5+4]
                    contain_prob = torch.FloatTensor([pred[i,j,b*5+4]])
                    xy = torch.FloatTensor([j,i])*cell_size #cell左上角  up left of cell
                    box[:2] = box[:2]*cell_size + xy # return cxcy relative to image
                    box_xy = torch.FloatTensor(box.size())#转换成xy形式    convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5*box[2:]
                    box_xy[2:] = box[:2] + 0.5*box[2:]
                    max_prob,cls_index = torch.max(pred[i,j,10:],0)
                    boxes.append(box_xy.view(1,4))
                    cls_indexs.append(cls_index)
                    probs.append(contain_prob)
    boxes = torch.cat(boxes,0) #(n,4)
    probs = torch.cat(probs,0) #(n,)
    cls_indexs = torch.cat(cls_indexs,0) #(n,)
    keep = nms(boxes,probs)
    return boxes[keep],cls_indexs[keep],probs[keep]

def nms(bboxes,scores,threshold=0.5):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1) * (y2-y1)

    _,order = scores.sort(0,descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)


def  eval():
    '''
    验证集 使用voc2012训练集去验证
    '''
    # efaultdict类就好像是一个dict字典，使用list类型来初始化
    target = defaultdict(list)
    preds = defaultdict(list)

    image_list = []  # image path list
    f = open(opt.voc_2012train)
    lines = f.readlines()
    file_list = []
    for line in lines:
        splited = line.strip().split()
        file_list.append(splited)
    f.close()
    print('---prepare target---')
    for image_file in tqdm(file_list):
        image_id = image_file[0]
        image_list.append(image_id)
        num_obj = int(image_file[1])
        for i in range(num_obj):
            x1 = int(image_file[2+5*i])
            y1 = int(image_file[3+5*i])
            x2 = int(image_file[4+5*i])
            y2 = int(image_file[5+5*i])
            c = int(image_file[6+5*i])
            class_name = opt.VOC_CLASSES[c]
            target[(image_id,class_name)].append([x1,y1,x2,y2])

    print('---start test---')
    model = vgg16(pretrained=False)
    model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(4096, 1470),
        )

    model.load_state_dict(torch.load(opt.best_test_loss_model_path))
    model.eval()
    if opt.use_gpu:
        model.cuda()
    count = 0
    for image_path in tqdm(image_list):
        result = predict_gpu(model, image_path,root_path=opt.file_root)  # result[[left_up,right_bottom,class_name,image_path],]
        for (x1, y1), (x2, y2), class_name, image_id, prob in result:  # image_id is actually image_path
                preds[class_name].append([image_id, prob, x1, y1, x2, y2])

    print('\n---start evaluate---')
    voc_eval(preds, target, VOC_CLASSES=opt.VOC_CLASSES)


def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.

    else:
        # correct ap caculation
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def voc_eval(preds, target, VOC_CLASSES=opt.VOC_CLASSES, threshold=0.5, use_07_metric=False, ):
    '''
    preds {'cat':[[image_id,confidence,x1,y1,x2,y2],...],'dog':[[],...]}
    target {(image_id,class):[[],]}

    举例：
    preds = {
        'cat': [['image01', 0.9, 20, 20, 40, 40], ['image01', 0.8, 20, 20, 50, 50], ['image02', 0.8, 30, 30, 50, 50]],
        'dog': [['image01', 0.78, 60, 60, 90, 90]]}
    target = {('image01', 'cat'): [[20, 20, 41, 41]], ('image01', 'dog'): [[60, 60, 91, 91]],
              ('image02', 'cat'): [[30, 30, 51, 51]]}
    '''
    aps = []
    for i, class_ in enumerate(VOC_CLASSES):
        pred = preds[class_]  # [[image_id,confidence,x1,y1,x2,y2],...]
        if len(pred) == 0:  # 如果这个类别一个都没有检测到的异常情况
            ap = -1
            print('---class {} ap {}---'.format(class_, ap))
            aps += [ap]
            break
        # print(pred)
        image_ids = [x[0] for x in pred]
        confidence = np.array([float(x[1]) for x in pred])
        BB = np.array([x[2:] for x in pred])
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        npos = 0.
        for (key1, key2) in target:
            if key2 == class_:
                npos += len(target[(key1, key2)])  # 统计这个类别的正样本，在这里统计才不会遗漏
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d, image_id in enumerate(image_ids):
            bb = BB[d]  # 预测框
            if (image_id, class_) in target:
                BBGT = target[(image_id, class_)]  # [[],]
                for bbgt in BBGT:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(bbgt[0], bb[0])
                    iymin = np.maximum(bbgt[1], bb[1])
                    ixmax = np.minimum(bbgt[2], bb[2])
                    iymax = np.minimum(bbgt[3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    union = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + (bbgt[2] - bbgt[0] + 1.) * (
                                bbgt[3] - bbgt[1] + 1.) - inters
                    if union == 0:
                        print(bb, bbgt)

                    overlaps = inters / union
                    if overlaps > threshold:
                        tp[d] = 1
                        BBGT.remove(bbgt)  # 这个框已经匹配到了，不能再匹配
                        if len(BBGT) == 0:
                            del target[(image_id, class_)]  # 删除没有box的键值
                        break
                fp[d] = 1 - tp[d]
            else:
                fp[d] = 1
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        # print(rec,prec)
        ap = voc_ap(rec, prec, use_07_metric)
        print('---class {} ap {}---'.format(class_, ap))
        aps += [ap]
    print('---map {}---'.format(np.mean(aps)))







# 主函数
if __name__ == '__main__':
    # eval()
    # train()
    predict()



