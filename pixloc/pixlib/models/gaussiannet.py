"""
A dummy model that computes an image pyramid with appropriate blurring.
"""

import torch
import kornia

from .base_model import BaseModel


class GaussianNet(BaseModel):
    default_conf = {
        'output_scales': [1, 4, 16],  # what scales to adapt and output
        'kernel_size_factor': 3,
    }

    def _init(self, conf):
        self.scales = conf['output_scales']

    def _forward(self, data):
        image = data['image']
        scale_prev = 1
        pyramid = []
        for scale in self.conf.output_scales:
            sigma = scale / scale_prev
            ksize = self.conf.kernel_size_factor * sigma
            image = kornia.filter.gaussian_blur2d(
                    image, kernel_size=ksize, sigma=sigma)
            if sigma != 1:
                image = torch.nn.functional.interpolate(
                    image, scale_factor=1/sigma, mode='bilinear',
                    align_corners=False)
            pyramid.append(image)
            scale_prev = scale
        return {'feature_maps': pyramid}

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError
