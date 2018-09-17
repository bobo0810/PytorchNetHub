import torch
import torch.nn.functional as F

from dice_loss import dice_coeff


def eval_net(net, dataset, gpu=False):
    '''
    :param net: 训练的网络
    :param dataset: 验证集
    '''
    """Evaluation without the densecrf with the dice coefficient"""
    tot = 0
    for i, b in enumerate(dataset):
        img = b[0]
        true_mask = b[1]

        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)

        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        mask_pred = net(img)[0]
        mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
        # 评价函数：Dice系数   Dice距离用于度量两个集合的相似性
        tot += dice_coeff(mask_pred, true_mask).item()
    return tot / i
