from __future__ import division
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_classes(path):
    """
    Loads class labels at 'path'
    加载数据集类别标签
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names

def weights_init_normal(m):
    '''
    初始化权重
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area =    torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                    torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1,  keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
    return output

def build_targets(pred_boxes, target, anchors, num_anchors, num_classes, dim, ignore_thres, img_dim):
    nB = target.size(0)  #batch个数  16
    nA = num_anchors     #锚框个数   3
    nC = num_classes     #数据集类别数  80
    dim = dim            #feature map相对于原图的缩放倍数13

    # 初始化参数
    mask        = torch.zeros(nB, nA, dim, dim)     #[16,3,13,13]   全0
    conf_mask   = torch.ones(nB, nA, dim, dim)      #[16,3,13,13]   全1
    tx          = torch.zeros(nB, nA, dim, dim)     #[16,3,13,13]   全0
    ty          = torch.zeros(nB, nA, dim, dim)     #[16,3,13,13]   全0
    tw          = torch.zeros(nB, nA, dim, dim)     #[16,3,13,13]   全0
    th          = torch.zeros(nB, nA, dim, dim)     #[16,3,13,13]   全0
    tconf       = torch.zeros(nB, nA, dim, dim)     #[16,3,13,13]   全0
    tcls        = torch.zeros(nB, nA, dim, dim, num_classes)    #[16,3,13,13,80]  全0

    # 为了计算一个batch中的recall召回率
    nGT = 0  # 统计 真值框个数 GT ground truth
    nCorrect = 0  # 统计 预测出有物体的个数 （即 真值框 与 3个原始锚框与真值框iou最大的那个锚框对应的预测框  之间的iou > 0.5 为预测正确）

    # 遍历每一张图片
    for b in range(nB):
        #遍历一张图片的所有物体
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0:
                # 即代表遍历完所有物体，continue直接开始下一次for循环(译者：使用break直接结束for循环更好)
                continue
            nGT += 1
            # Convert to position relative to box
            # target真值框 坐标被归一化后[16,50,5] 值在0-1之间。故乘以 dim  将尺度转化为  13x13尺度下的真值框
            gx = target[b, t, 1] * dim
            gy = target[b, t, 2] * dim
            gw = target[b, t, 3] * dim
            gh = target[b, t, 4] * dim
            # Get grid box indices 向下取整，获取网格框索引，即左上角偏移坐标
            gi = int(gx)
            gj = int(gy)
            # Get shape of gt box [1,4]
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # Get shape of anchor box [3,4]   前两列全为0  后两列为 三个anchor的w、h
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # Calculate iou between gt and anchor shapes
            # 计算 一个真值框 与  对应的3个原始锚框  之间的iou
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            # Where the overlap is larger than threshold set mask to zero (ignore)   当iou重叠率>阈值，则置为0
            # conf_mask全为1 [16,3,13,13]  当一个真值框 与  一个原始锚框  之间的iou > 阈值时，则置为0。
            # 即 将 负责预测物体的网格及 它周围的网格 都置为0 不参与训练，后面的代码会 将负责预测物体的网格再置为1。
            conf_mask[b, anch_ious > ignore_thres] = 0
            # Find the best matching anchor box  找到 一个真值框 与  对应的3个原始锚框  之间的iou最大的  下标值
            best_n = np.argmax(anch_ious)
            # Get ground truth box [1,4]
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            # Get the best prediction  [1,4]
            # pred_boxes:在13x13尺度上的预测框
            # pred_box：取出  3个原始锚框与 真值框 iou最大的那个锚框  对应的预测框
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            # Masks   [16,3,13,13]   全0      在3个原始锚框与 真值框 iou最大的那个锚框  对应的预测框位，即 负责预测物体的网格置为1 （此时它周围网格为0，思想类似nms）
            mask[b, best_n, gj, gi] = 1
            #  [16,3,13,13]   全1 然后将 负责预测物体的网格及 它周围的网格 都置为0 不参与训练 ，然后  将负责预测物体的网格再次置为1。
            #  即总体思想为： 负责预测物体的网格 位置置为1，它周围的网格置为0。类似NMS 非极大值抑制
            conf_mask[b, best_n, gj, gi] = 1
            # Coordinates 坐标     gi= gx的向下取整。  gx-gi、gy-gj 为 网格内的 物体中心点坐标（0-1之间）
            # tx  ty初始化全为0，在有真值框的网格位置写入 真实的物体中心点坐标
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            #  论文中 13x13尺度下真值框=原始锚框 x 以e为底的 预测值。故预测值= log(13x13尺度下真值框  / 原始锚框  +  1e-16 )
            tw[b, best_n, gj, gi] = math.log(gw/anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh/anchors[best_n][1] + 1e-16)
            # One-hot encoding of label
            tcls[b, best_n, gj, gi, int(target[b, t, 0])] = 1
            # Calculate iou between ground truth and best matching prediction 计算真值框 与   3个原始锚框与真值框iou最大的那个锚框对应的预测框    之间的iou
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            # [16,3,13,13]   全0，有真值框对应的网格位置为1  标明 物体中心点落在该网格中，该网格去负责预测物体
            tconf[b, best_n, gj, gi] = 1

            if iou > 0.5:
                nCorrect += 1
    # nGT 统计一个batch中的真值框个数
    # nCorrect 统计 一个batch预测出有物体的个数
    # mask   [16,3,13,13] 初始化全0   在3个原始锚框与 真值框 iou最大的那个锚框  对应的预测框位置置为1
    # conf_mask  [16,3,13,13]  初始化全1，之后的操作：负责预测物体的网格置为1，它周围网格置为0
    # tx, ty [16,3,13,13] 初始化全为0，在有真值框的网格位置写入 真实的物体中心点坐标
    # tw, th  [16,3,13,13] 初始化全为0，该值为 真值框的w、h 按照公式转化为 网络输出时对应的真值（该值对应于 网络输出的真值）
    # tconf [16,3,13,13]   初始化全0，有真值框对应的网格位置为1  标明 物体中心点落在该网格中，该网格去负责预测物体
    # tcls    #[16,3,13,13,80]  初始化全0，经过one-hot编码后  在真实类别处值为1
    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.from_numpy(np.eye(num_classes, dtype='uint8')[y])
