from typing import Dict, Union
from omegaconf import DictConfig, OmegaConf as oc
import numpy as np
import torch

from ..pixlib.datasets.view import resize, numpy_image_to_torch


class FeatureExtractor(torch.nn.Module):
    default_conf: Dict = dict(
        resize=1024,
        resize_by='max',
    )

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 conf: Union[Dict, DictConfig]):
        super().__init__()
        self.conf = oc.merge(oc.create(self.default_conf), oc.create(conf))
        self.device = device
        self.model = model

        assert hasattr(self.model, 'scales')
        assert self.conf.resize_by in ['max', 'max_force'], self.conf.resize_by
        self.to(device)
        self.eval()

    def prepare_input(self, image: np.array) -> torch.Tensor:
        return numpy_image_to_torch(image).to(self.device).unsqueeze(0)

    @torch.no_grad()
    def __call__(self, image: np.array, scale_image: int = 1):
        """Extract feature-maps for a given image.
        Args:
            image: input image (H, W, C)
        """
        image = image.astype(np.float32)  # better for resizing
        scale_resize = (1., 1.)
        if self.conf.resize is not None:
            target_size = self.conf.resize // scale_image
            if (max(image.shape[:2]) > target_size or
                    self.conf.resize_by == 'max_force'):
                image, scale_resize = resize(image, target_size, max, 'linear')

        image_tensor = self.prepare_input(image)
        pred = self.model({'image': image_tensor})
        features = pred['feature_maps']
        assert len(self.model.scales) == len(features)

        features = [feat.squeeze(0) for feat in features]  # remove batch dim
        confidences = pred.get('confidences')
        if confidences is not None:
            confidences = [c.squeeze(0) for c in confidences]

        scales = [(scale_resize[0]/s, scale_resize[1]/s)
                  for s in self.model.scales]

        return features, scales, confidences
