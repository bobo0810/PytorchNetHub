import os

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.autograd import Variable
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

#RLIMIT_NOFILE一个进程能打开的最大文件数，内核默认是1024
#rlimit的两个参数
#soft limit  指内核所能支持的资源上限 最大也只能达到1024
#hard limit 学校机房值为4096  在资源中只是作为soft limit的上限。当你设置hard limit后，你以后设置的soft limit只能小于hard limit。

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
#图形并没有在屏幕上显示,但是已保存到文件,关键是要设置'Agg'的属性
matplotlib.use('agg')



def eval(dataloader, faster_rcnn, test_num=10000):
    """
    验证
    """
    #空的list
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0], sizes[1][0]]
        #预测目标框、类别、分数存入list
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        #将真实的目标框、类别、difficults存入list
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        #将预测的目标框、类别、分数存入list
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break
    #返回dictz字典，两个key值：AP、mAP
    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    """
    训练
    """
    #解析命令行参数，设置配置文件参数
    opt._parse(kwargs)
    #初始化Dataset参数
    dataset = Dataset(opt)
    print('load data')
    #data_ 数据加载器（被重命名，pytorch方法）
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    #初始化TestDataset参数
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    #新建一个FasterRCNNVGG16
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    #新建一个trainer，并将网络模型转移到GPU上
    #将FasterRCNNVGG16模型传入
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    #如果存在，加载训练好的模型
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    #可视化类别 vis为visdom加载器
    trainer.vis.text(dataset.db.label_names, win='labels')
    #best_map存放的是 最优的mAP的网络参数
    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        #trainer方法 将平均精度的元组 和 混淆矩阵的值置0
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            #调整数据的形状    scale：缩放倍数（输入图片尺寸 比上 输出数据的尺寸）
            #1.6左右 供模型训练之前将模型规范化
			scale = at.scalar(scale)
            #将数据集转入到GPU上
			#img  1x3x800x600  一张图片 三通道  大小800x600（不确定）
			#bbox 1x1x4
			#label 1x1
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            #将数据转为V 变量，以便进行自动反向传播
            img, bbox, label = Variable(img), Variable(bbox), Variable(label)
            #训练并更新可学习参数（重点*****）  前向+反向，返回losses
            trainer.train_step(img, bbox, label, scale)
            #进行多个数据的可视化
            if (ii + 1) % opt.plot_every == 0:
                #进入调试模式
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss  画五个损失
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes  img[0]，是压缩0位，形状变为[3x800x600]
                #反向归一化，将img反向还原为原始图像，以便用于显示
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                #通过原始图像，真实bbox，真实类别 进行显示
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)

                # plot predicti bboxes
                #对原图进行预测，得到预测的bbox  label  scores
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                #通过原始图像、预测的bbox，预测的类别   以及概率  进行显示
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                #rpn混淆矩阵
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                #roi混淆矩阵
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        #使用验证集对当前的网络进行验证，返回一个字典，key值有AP,mAP
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        #如果当前的map值优于best_map，则将当前值赋给best_map。将当前模型保留
        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        #如果epoch到达9时，加载 当前的最优模型，并将学习率按lr_decay衰减调低
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay
        #可视化验证集的test_map 和log信息
        trainer.vis.plot('test_map', eval_result['map'])
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        trainer.vis.log(log_info)
        if epoch == 13: 
            break


if __name__ == '__main__':
    import fire

    fire.Fire()
