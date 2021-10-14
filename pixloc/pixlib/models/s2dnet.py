"""
An implementation of
    S2DNet: Learning Image Features for Accurate Sparse-to-Dense Matching
    Hugo Germain, Guillaume Bourmaud, Vincent Lepetit
    European Conference on Computer Vision (ECCV) 2020

Adapted from https://github.com/germain-hug/S2DNet-Minimal
"""

from typing import List
import torch
import torch.nn as nn
from torchvision import models
import logging

from .base_model import BaseModel
from ...settings import DATA_PATH

logger = logging.getLogger(__name__)


# VGG-16 Layer Names and Channels
vgg16_layers = {
    "conv1_1": 64,
    "relu1_1": 64,
    "conv1_2": 64,
    "relu1_2": 64,
    "pool1": 64,
    "conv2_1": 128,
    "relu2_1": 128,
    "conv2_2": 128,
    "relu2_2": 128,
    "pool2": 128,
    "conv3_1": 256,
    "relu3_1": 256,
    "conv3_2": 256,
    "relu3_2": 256,
    "conv3_3": 256,
    "relu3_3": 256,
    "pool3": 256,
    "conv4_1": 512,
    "relu4_1": 512,
    "conv4_2": 512,
    "relu4_2": 512,
    "conv4_3": 512,
    "relu4_3": 512,
    "pool4": 512,
    "conv5_1": 512,
    "relu5_1": 512,
    "conv5_2": 512,
    "relu5_2": 512,
    "conv5_3": 512,
    "relu5_3": 512,
    "pool5": 512,
}


class AdapLayers(nn.Module):
    """Small adaptation layers.
    """

    def __init__(self, hypercolumn_layers: List[str], output_dim: int = 128):
        """Initialize one adaptation layer for every extraction point.

        Args:
            hypercolumn_layers: The list of the hypercolumn layer names.
            output_dim: The output channel dimension.
        """
        super(AdapLayers, self).__init__()
        self.layers = []
        channel_sizes = [vgg16_layers[name] for name in hypercolumn_layers]
        for i, l in enumerate(channel_sizes):
            layer = nn.Sequential(
                nn.Conv2d(l, 64, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, output_dim, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(output_dim),
            )
            self.layers.append(layer)
            self.add_module("adap_layer_{}".format(i), layer)

    def forward(self, features: List[torch.tensor]):
        """Apply adaptation layers.
        """
        for i, _ in enumerate(features):
            features[i] = getattr(self, "adap_layer_{}".format(i))(features[i])
        return features


class S2DNet(BaseModel):
    default_conf = {
        'hypercolumn_layers': ["conv1_2", "conv3_3", "conv5_3"],
        'checkpointing': None,
        'output_dim': 128,
        'pretrained': 's2dnet',
    }
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def _init(self, conf):
        assert conf.pretrained in ['s2dnet', 'imagenet', None]

        self.layer_to_index = {k: v for v, k in enumerate(vgg16_layers.keys())}
        self.hypercolumn_indices = [
                self.layer_to_index[n] for n in conf.hypercolumn_layers]
        num_layers = self.hypercolumn_indices[-1] + 1

        # Initialize architecture
        vgg16 = models.vgg16(pretrained=conf.pretrained == 'imagenet')
        layers = list(vgg16.features.children())[:num_layers]
        self.encoder = nn.ModuleList(layers)

        self.scales = []
        current_scale = 0
        for i, layer in enumerate(layers):
            if isinstance(layer, torch.nn.MaxPool2d):
                current_scale += 1
            if i in self.hypercolumn_indices:
                self.scales.append(2**current_scale)

        self.adaptation_layers = AdapLayers(
                conf.hypercolumn_layers, conf.output_dim)

        if conf.pretrained == 's2dnet':
            path = DATA_PATH / 's2dnet_weights.pth'
            logger.info(f'Loading S2DNet checkpoint at {path}.')
            state_dict = torch.load(path, map_location='cpu')['state_dict']
            params = self.state_dict()
            state_dict = {k: v for k, v in state_dict.items()
                          if v.shape == params[k].shape}
            self.load_state_dict(state_dict, strict=False)

    def _forward(self, data):
        image = data['image']
        mean, std = image.new_tensor(self.mean), image.new_tensor(self.std)
        image = (image - mean[:, None, None]) / std[:, None, None]

        feature_map = image
        feature_maps = []
        start = 0
        for idx in self.hypercolumn_indices:
            if self.conf.checkpointing:
                blocks = list(range(start, idx+2, self.conf.checkpointing))
                if blocks[-1] != idx+1:
                    blocks.append(idx+1)
                for start_, end_ in zip(blocks[:-1], blocks[1:]):
                    feature_map = torch.utils.checkpoint.checkpoint(
                        nn.Sequential(*self.encoder[start_:end_]), feature_map)
            else:
                for i in range(start, idx + 1):
                    feature_map = self.encoder[i](feature_map)
            feature_maps.append(feature_map)
            start = idx + 1

        feature_maps = self.adaptation_layers(feature_maps)
        return {'feature_maps': feature_maps}

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError
