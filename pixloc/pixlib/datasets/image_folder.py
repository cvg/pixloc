"""
Simply load images from a folder or nested folders (does not have any split).
"""

from pathlib import Path
import torch
import cv2
import numpy as np
import logging
import omegaconf

from .base_dataset import BaseDataset
from .utils.preprocessing import resize, numpy_image_to_torch


class ImageFolder(BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        'glob': ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'],
        'grayscale': False,
        'images': '???',
        'resize': None,
        'resize_by': 'max',
        'interpolation': 'linear',
        'root_folder': '/',
    }

    def _init(self, conf):
        self.root = conf.root_folder
        if isinstance(conf.images, str):
            if not Path(conf.images).is_dir():
                with open(conf.images, 'r') as f:
                    self.images = f.read().rstrip('\n').split('\n')
                logging.info(f'Found {len(self.images)} images in list file.')
            else:
                self.images = []
                glob = [conf.glob] if isinstance(conf.glob, str) else conf.glob
                for g in glob:
                    self.images += list(Path(conf.images).glob('**/'+g))
                if len(self.images) == 0:
                    raise ValueError(
                        f'Could not find any image in folder: {conf.images}.')
                self.images = [i.relative_to(conf.images) for i in self.images]
                self.root = conf.images
                logging.info(f'Found {len(self.images)} images in folder.')
        elif isinstance(conf.images, omegaconf.listconfig.ListConfig):
            self.images = conf.images.to_container()
        else:
            raise ValueError(conf.images)

    def get_dataset(self, split):
        return self

    def __getitem__(self, idx):
        path = self.images[idx]
        if self.conf.grayscale:
            mode = cv2.IMREAD_GRAYSCALE
        else:
            mode = cv2.IMREAD_COLOR
        img = cv2.imread(str(Path(self.root, path)), mode)
        if img is None:
            logging.warning(f'Image {str(path)} could not be read.')
            img = np.zeros((1024, 1024)+(() if self.conf.grayscale else (3,)))
        img = img.astype(np.float32)
        size = img.shape[:2][::-1]

        if self.conf.resize:
            args = {'interp': self.conf.interpolation}
            h, w = img.shape[:2]
            if self.conf.resize_by in ['max', 'force-max']:
                if ((self.conf.resize_by == 'force-max') or
                        (max(h, w) > self.conf.resize)):
                    img, _ = resize(img, self.conf.resize, fn=max, **args)
            elif self.conf.resize_by == 'min':
                if min(h, w) < self.conf.resize:
                    img, _ = resize(img, self.conf.resize, fn=min, **args)
            else:
                img, _ = resize(img, self.conf.resize, **args)

        data = {
            'name': str(path),
            'image': numpy_image_to_torch(img),
            'original_image_size': np.array(size),
        }
        return data

    def __len__(self):
        return len(self.images)
