#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torchvision.transforms.functional as tvF
from torch.utils.data import Dataset, DataLoader

import os
from sys import platform
import numpy as np
import random
from string import ascii_letters
from PIL import Image, ImageFont, ImageDraw

def load_dataset(root_dir, redux, params, shuffled=False, single=False):
    """
    Loads dataset and returns corresponding data loader.  数据集加载器
     redux指 训练集大小 or 验证集大小
    """

    # Create Torch dataset  新增 数据集   噪声类型和噪声参数
    noise = (params.noise_type, params.noise_param)

    # Instantiate appropriate dataset class  实例化数据集
    # if params.noise_type == 'mc':
    # #mc指 蒙特卡洛
    #     dataset = MonteCarloDataset(root_dir, redux, params.crop_size,
    #         clean_targets=params.clean_targets)
    # else:
    # 其余噪声 生成数据集
    dataset = NoisyDataset(root_dir, redux, params.crop_size,
        clean_targets=params.clean_targets, noise_dist=noise, seed=params.seed)

    # Use batch size of 1, if requested (e.g. test set)
    # 当single=True,则 batch_size=1 （测试集需要）
    if single:
        return DataLoader(dataset, batch_size=1, shuffle=shuffled)
    else:
        return DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffled)


class AbstractDataset(Dataset):
    """Abstract dataset class for Noise2Noise.  Noise2Noise的抽象数据集类"""

    def __init__(self, root_dir, redux=0, crop_size=128, clean_targets=False):
        """Initializes abstract dataset."""

        super(AbstractDataset, self).__init__()

        self.imgs = []
        self.root_dir = root_dir
        self.redux = redux
        self.crop_size = crop_size
        self.clean_targets = clean_targets

    def _random_crop(self, img_list):
        """Performs random square crop of fixed size.
        Works with list so that all items get the same cropped window (e.g. for buffers).
        """

        w, h = img_list[0].size
        assert w >= self.crop_size and h >= self.crop_size, \
            f'Error: Crop size: {self.crop_size}, Image size: ({w}, {h})'
        cropped_imgs = []
        i = np.random.randint(0, h - self.crop_size + 1)
        j = np.random.randint(0, w - self.crop_size + 1)

        for img in img_list:
            # Resize if dimensions are too small
            if min(w, h) < self.crop_size:
                img = tvF.resize(img, (self.crop_size, self.crop_size))

            # Random crop
            cropped_imgs.append(tvF.crop(img, i, j, self.crop_size, self.crop_size))

        return cropped_imgs


    def __getitem__(self, index):
        """Retrieves image from data folder."""

        raise NotImplementedError('Abstract method not implemented!')


    def __len__(self):
        """Returns length of dataset."""

        return len(self.imgs)


class NoisyDataset(AbstractDataset):
    """Class for injecting random noise into dataset. 将随机噪声注入数据集，继承AbstractDataset类   """

    def __init__(self, root_dir, redux, crop_size, clean_targets=False,
        noise_dist=('gaussian', 50.), seed=None):
        """Initializes noisy image dataset.  初始化 噪声数据集"""

        super(NoisyDataset, self).__init__(root_dir, redux, crop_size, clean_targets)

        # 读取数据集
        self.imgs = os.listdir(root_dir)

        if redux:
            self.imgs = self.imgs[:redux]

        # Noise parameters (max std for Gaussian, lambda for Poisson, nb of artifacts for text)
        # 噪声参数
        self.noise_type = noise_dist[0]
        self.noise_param = noise_dist[1]
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)


    def _add_noise(self, img):
        """Adds Gaussian or Poisson noise to image."""

        w, h = img.size
        c = len(img.getbands())

        # Poisson distribution
        # It is unclear how the paper handles this. Poisson noise is not additive,
        # it is data dependent, meaning that adding sampled valued from a Poisson
        # will change the image intensity...
        if self.noise_type == 'poisson':
            noise = np.random.poisson(img)
            noise_img = img + noise
            noise_img = 255 * (noise_img / np.amax(noise_img))

        # Normal distribution (default)
        else:
            if self.seed:
                std = self.noise_param
            else:
                std = np.random.uniform(0, self.noise_param)
            noise = np.random.normal(0, std, (h, w, c))

            # Add noise and clip
            noise_img = np.array(img) + noise

        noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
        return Image.fromarray(noise_img)


    def _add_text_overlay(self, img):
        """Adds text overlay to images. 为图像添加文本叠加"""

        assert self.noise_param < 1, 'Text parameter is an occupancy probability'

        w, h = img.size
        c = len(img.getbands())

        # Choose font and get ready to draw
        if platform == 'linux':
            serif = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
        else:
            serif = 'Times New Roman.ttf'
        text_img = img.copy()
        text_draw = ImageDraw.Draw(text_img)

        # Text binary mask to compute occupancy efficiently
        w, h = img.size
        mask_img = Image.new('1', (w, h))
        mask_draw = ImageDraw.Draw(mask_img)

        # Random occupancy in range [0, p]
        if self.seed:
            random.seed(self.seed)
            max_occupancy = self.noise_param
        else:
            max_occupancy = np.random.uniform(0, self.noise_param)
        def get_occupancy(x):
            y = np.array(x, dtype=np.uint8)
            return np.sum(y) / y.size

        # Add text overlay by choosing random text, length, color and position
        while 1:
            font = ImageFont.truetype(serif, np.random.randint(16, 21))
            length = np.random.randint(10, 25)
            chars = ''.join(random.choice(ascii_letters) for i in range(length))
            color = tuple(np.random.randint(0, 255, c))
            pos = (np.random.randint(0, w), np.random.randint(0, h))
            text_draw.text(pos, chars, color, font=font)

            # Update mask and check occupancy
            mask_draw.text(pos, chars, 1, font=font)
            if get_occupancy(mask_img) > max_occupancy:
                break

        return text_img


    def _corrupt(self, img):
        """Corrupts images (Gaussian, Poisson, or text overlay)."""

        if self.noise_type in ['gaussian', 'poisson']:
            return self._add_noise(img)
        elif self.noise_type == 'text':
            return self._add_text_overlay(img)
        else:
            raise ValueError('Invalid noise type: {}'.format(self.noise_type))


    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        # Load PIL image
        img_path = os.path.join(self.root_dir, self.imgs[index])
        img =Image.open(img_path).convert('RGB')

        # Random square crop
        if self.crop_size != 0:
            img = self._random_crop([img])[0]

        # Corrupt source image 转化为损坏图像
        source = tvF.to_tensor(self._corrupt(img))

        # Corrupt target image, but not when clean targets are requested
        # 选择 标签为 干净图像 or 噪声图像
        if self.clean_targets:
            target = tvF.to_tensor(img)
        else:
            target = tvF.to_tensor(self._corrupt(img))

        return source, target

