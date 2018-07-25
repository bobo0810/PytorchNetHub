# -*- coding: utf-8 -*-
import torch


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.

    将预先生成的锚坐标由 中心点坐标和宽高 转化为 (xmin, ymin, xmax, ymax)形式
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    #按照纵坐标将结果合并.
    # 中心点坐标-（w,h）*（1/2）即为左上角坐标xmin, ymin
    # 中心点坐标+（w,h）*（1/2）即为右下角坐标xmax, ymax
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.

    计算两组盒子两两的jaccard重叠，即IOU重叠率。 jaccard重叠只是两个盒子的联合交叉。
    我们在此进行操作地面实况框和默认框
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    将每个预测框与最高IOU的真实框相匹配，对边界框进行编码，然后返回匹配的索引
  对应于置信度和位置预测。
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.  0.5
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].  真值框[一张图片对应的]
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].  锚框坐标[8732,4]
        variances: (tensor) Variances corresponding to each prior coord,  对应于每个锚框的方差 [8732,4]
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].   对于该图像中物体的类别标签真值[物体数]
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.  保存 编码后的回归目标
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.   保存 匹配好的
        idx: (int) current batch index   batch中当前第idx张图片
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
        匹配的索引对应于  1）位置  和  2）置信度。
    """
    # jaccard index
    #若truths[2,4] priors[8732,4]，则结果为[2,8732]
    #计算真值框与锚之间 两两的重叠率IOU
    overlaps = jaccard(
        truths,
        # 将预先生成的锚坐标由中心点坐标和宽高转化为(xmin, ymin, xmax, ymax)形式
        point_form(priors)
    )

    #原论文的匹配方法:
    #1、用 MultiBox 中的 best jaccard overlap 来匹配每一个 ground truth box 与 default box，这样就能保证每一个 groundtruth box 与唯一的一个 default box 对应起来
    #2、又不同于 MultiBox ，将 default box 与任何的 groundtruth box 配对，只要两者之间的 jaccard overlap 大于一个阈值，这里本文的阈值为 0.5

    # (Bipartite Matching)二分匹配.
    # [1,num_objects] best prior for each ground truth
    #对于每个真值框，找出 与其匹配IOU最高的锚框  （相对于overlaps，找出每一行最大的值，返回该值及对应的序号）
    #best_prior_overlap为 与真值框匹配IOU最高的值。best_prior_idx为  对应序号。 形状 [1,num_objects]
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    # 对于每一个锚框，找出 与其匹配IOU最高的真值框  （相对于overlaps，找出每一列最大的值，返回该值及对应的序号）
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    # 压缩tensor
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    #pytorch方法  index_fill_(dim, index, val)
    # 按参数 index 给出的索引序列, 将原 tensor 中的元素用 val 填充
    #dim (int) – 索引 index 所指向的维度
    #index (LongTensor) – 从参数 val 中选取数据的索引序列
    #即将best_truth_overlap矩阵中 按照每列中，序号包含在best_prior_idx中的值都被替换为2
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    # 确保每个真值框 匹配到与其IOU最大的锚
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # 对于每个锚框，匹配到的真值框的坐标Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    # 得到一张图的分类误差
    conf[best_truth_overlap < threshold] = 0  # label as background 将IOU小于阈值的标签置为0,，即背景
    #得到一张图的真值框与锚的坐标偏移值[8732, 4]
    loc = encode(matches, priors, variances)
    #保存一个batch中的坐标偏移值  和 分类误差
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn  编码匹配到的真值框与锚的坐标偏移，以便网络学习
    conf_t[idx] = conf  # [num_priors] top class label for each prior   对于每个锚框的前几个类别标签


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.

    将 来自锚框层的方差编码到  与这些锚框相匹配到的 真值框中
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form （对于每一个锚框，都匹配一个IOU大于阈值的真值框）
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form （值为中心偏移形式的锚框）
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes （锚框的方差）
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    #原论文中Lloc 即定位损失计算公式
    # dist b/t match center and prior's center
    #即为 对于每一个锚框，都匹配一个IOU大于阈值的真值框中心点坐标- 公式生成的锚框
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance编码方差
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh  计算 wh/ 锚的 wh,之后再log
    #xmax-xmin=w   ymax-ymin=h
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.

    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    # 1、矩阵逐元素-矩阵最大值
    # 2、矩阵逐元素求exp
    # 3、每行相加，结果 行不变，列变为1
    # 4、逐元素log
    # 5、逐元素+矩阵最大值
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
