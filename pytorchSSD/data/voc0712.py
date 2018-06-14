"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# note: if you used our download scripts, this should be right
VOC_ROOT = osp.join(HOME, "data/VOCdevkit/")


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    将VOC注释转换为真值框bbox坐标 和 标签索引的tensor
  用索引的字典名称的dict查找

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
            一个包含bbox坐标 和 类名的 list
        """
        res = []
        # 遍历这张图的所有物体
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # 形状形如[[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object
    VOC数据集，继承data.Dataset，需实现__getitem__、__len__方法

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
                       数据集根目录
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
                        训练or 验证or 测试
        transform (callable, optional): transformation to perform on the
            input image
                          图像转化和数据增强
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform   #SSDAugmentation(cfg['min_dim'],MEANS))  图像增强
        self.target_transform = target_transform   #VOCAnnotationTransform()  注释变换
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')  #读取所有xml文件
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')    #读取所有jpg文件
        self.ids = list()   #图像的id全部保存在ids
        # 2007和2012的训练验证集
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        '''
        :param index: 取第几条数据
        :return: 一张图像及对应的真值框和类别
        '''
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        '''
        取某条数据
        :param index: 取第几条数据
        :return: 一张图像、对应的真值框、高、宽
        '''
        img_id = self.ids[index]
        # 读取图像与对应的xml
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        # 得到图像的高、宽、通道（数据集中高宽不一定）
        height, width, channels = img.shape
        # VOCAnnotationTransform()  注释变换（解析xml，返回一个list  包含所有对象的bbox坐标与类名）
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        # SSDAugmentation(cfg['min_dim'],MEANS))  图像增强
        if self.transform is not None:
            # 转化为tensor  形状为（x,5）  x:图像中的物体总数   5：bbox坐标、 类别
            target = np.array(target)
            # 图像增强
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb  转化为rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            # hstack合并  axis=1按照列合并   target：一行内容是boxes坐标+类别
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        # 一张图像、对应的真值框和类别、高、宽
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
