#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    # eg：f[:-4]是为了去掉 .jpg 后缀。结果只为 照片名称，无后缀。
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i) for i in range(n) for id in ids)


def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    #返回  tuples，（img的resize后的tensor,序号）
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        # get_square: 当pos为0 时，裁剪宽度，得到左边部分图片[384,384,3]   当pos为1 时，裁剪宽度，得到右边部分图片[384,190,3]
        yield get_square(im, pos)

def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    '''
    :param ids:
    :param dir_img: 图片路径
    :param dir_mask: mask图片路径
    :param scale: 图像训练时缩小倍数
    :return:all the couples (img, mask)
    '''
    """Return all the couples (img, mask)"""

    # 读取图片，并按照scale进行resize
    imgs = to_cropped_imgs(ids, dir_img, '.jpg', scale)

    # need to transform from HWC to CHW  转化（高H、宽W、通道C）为（通道C、高H、宽W）
    imgs_switched = map(hwc_to_chw, imgs)
    # 归一化（值转化到0-1之间）
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask, '_mask.gif', scale)
    # list( rezise且经过转化和归一化后的图像tensor,resize后的mask图像tensor)
    return zip(imgs_normalized, masks)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '_mask.gif')
    return np.array(im), np.array(mask)
