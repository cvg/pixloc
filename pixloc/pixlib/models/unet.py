"""
Flexible UNet model which takes any Torchvision backbone as encoder.
Predicts multi-level feature and uncertainty maps
and makes sure that they are well aligned.
"""

import torchvision
import torch
import torch.nn as nn

from .base_model import BaseModel
from .utils import checkpointed


class DecoderBlock(nn.Module):
    def __init__(self, previous, skip, out, num_convs=1, norm=nn.BatchNorm2d):
        super().__init__()

        self.upsample = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False)

        layers = []
        for i in range(num_convs):
            conv = nn.Conv2d(
                previous+skip if i == 0 else out, out,
                kernel_size=3, padding=1, bias=norm is None)
            layers.append(conv)
            if norm is not None:
                layers.append(norm(out))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, previous, skip):
        upsampled = self.upsample(previous)
        # If the shape of the input map `skip` is not a multiple of 2,
        # it will not match the shape of the upsampled map `upsampled`.
        # If the downsampling uses ceil_mode=False, we nedd to crop `skip`.
        # If it uses ceil_mode=True (not supported here), we should pad it.
        _, _, hu, wu = upsampled.shape
        _, _, hs, ws = skip.shape
        assert (hu <= hs) and (wu <= ws), 'Using ceil_mode=True in pooling?'
        # assert (hu == hs) and (wu == ws), 'Careful about padding'
        skip = skip[:, :, :hu, :wu]
        return self.layers(torch.cat([upsampled, skip], dim=1))


class AdaptationBlock(nn.Sequential):
    def __init__(self, inp, out):
        conv = nn.Conv2d(inp, out, kernel_size=1, padding=0, bias=True)
        super().__init__(conv)


class UNet(BaseModel):
    default_conf = {
        'output_scales': [0, 2, 4],  # what scales to adapt and output
        'output_dim': 128,  # # of channels in output feature maps
        'encoder': 'vgg16',  # string (torchvision net) or list of channels
        'num_downsample': 4,  # how many downsample block (if VGG-style net)
        'decoder': [64, 64, 64, 64],  # list of channels of decoder
        'decoder_norm': 'nn.BatchNorm2d',  # normalization ind decoder blocks
        'do_average_pooling': False,
        'compute_uncertainty': False,
        'checkpointed': False,  # whether to use gradient checkpointing
    }
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def build_encoder(self, conf):
        assert isinstance(conf.encoder, str)
        Encoder = getattr(torchvision.models, conf.encoder)
        encoder = Encoder(pretrained=True)
        Block = checkpointed(torch.nn.Sequential, do=conf.checkpointed)

        if conf.encoder.startswith('vgg'):
            # Parse the layers and pack them into downsampling blocks
            # It's easy for VGG-style nets because of their linear structure.
            # This does not handle strided convs and residual connections
            assert max(conf.output_scales) <= conf.num_downsample
            skip_dims = []
            previous_dim = None
            blocks = [[]]
            for i, layer in enumerate(encoder.features):
                if isinstance(layer, torch.nn.Conv2d):
                    previous_dim = layer.out_channels
                elif isinstance(layer, torch.nn.MaxPool2d):
                    assert previous_dim is not None
                    skip_dims.append(previous_dim)
                    if (conf.num_downsample + 1) == len(blocks):
                        break
                    blocks.append([])
                    if conf.do_average_pooling:
                        assert layer.dilation == 1
                        layer = torch.nn.AvgPool2d(
                            kernel_size=layer.kernel_size, stride=layer.stride,
                            padding=layer.padding, ceil_mode=layer.ceil_mode,
                            count_include_pad=False)
                blocks[-1].append(layer)
            assert (conf.num_downsample + 1) == len(blocks)
            encoder = [Block(*b) for b in blocks]
        elif conf.encoder.startswith('resnet'):
            # Manually define the splits - this could be improved
            assert conf.encoder[len('resnet'):] in ['18', '34', '50', '101']
            block1 = torch.nn.Sequential(encoder.conv1, encoder.bn1,
                                         encoder.relu)
            block2 = torch.nn.Sequential(encoder.maxpool, encoder.layer1)
            block3 = encoder.layer2
            block4 = encoder.layer3
            blocks = [block1, block2, block3, block4]
            encoder = [torch.nn.Identity()] + [Block(b) for b in blocks]
            skip_dims = [3, 64, 256, 512, 1024]
        else:
            raise NotImplementedError(conf.encoder)

        encoder = nn.ModuleList(encoder)
        return encoder, skip_dims

    def _init(self, conf):
        # Encoder
        self.encoder, skip_dims = self.build_encoder(conf)

        # Decoder
        if conf.decoder is not None:
            assert len(conf.decoder) == (len(skip_dims) - 1)
            Block = checkpointed(DecoderBlock, do=conf.checkpointed)
            norm = eval(conf.decoder_norm) if conf.decoder_norm else None

            previous = skip_dims[-1]
            decoder = []
            for out, skip in zip(conf.decoder, skip_dims[:-1][::-1]):
                decoder.append(Block(previous, skip, out, norm=norm))
                previous = out
            self.decoder = nn.ModuleList(decoder)

        # Adaptation layers
        adaptation = []
        if conf.compute_uncertainty:
            uncertainty = []
        for idx, i in enumerate(conf.output_scales):
            if conf.decoder is None or i == (len(self.encoder) - 1):
                input_ = skip_dims[i]
            else:
                input_ = conf.decoder[-1-i]

            # out_dim can be an int (same for all scales) or a list (per scale)
            dim = conf.output_dim
            if not isinstance(dim, int):
                dim = dim[idx]

            block = AdaptationBlock(input_, dim)
            adaptation.append(block)
            if conf.compute_uncertainty:
                uncertainty.append(AdaptationBlock(input_, 1))
        self.adaptation = nn.ModuleList(adaptation)
        self.scales = [2**s for s in conf.output_scales]
        if conf.compute_uncertainty:
            self.uncertainty = nn.ModuleList(uncertainty)

    def _forward(self, data):
        image = data['image']
        mean, std = image.new_tensor(self.mean), image.new_tensor(self.std)
        image = (image - mean[:, None, None]) / std[:, None, None]

        skip_features = []
        features = image
        for block in self.encoder:
            features = block(features)
            skip_features.append(features)

        if self.conf.decoder:
            pre_features = [skip_features[-1]]
            for block, skip in zip(self.decoder, skip_features[:-1][::-1]):
                pre_features.append(block(pre_features[-1], skip))
            pre_features = pre_features[::-1]  # fine to coarse
        else:
            pre_features = skip_features

        out_features = []
        for adapt, i in zip(self.adaptation, self.conf.output_scales):
            out_features.append(adapt(pre_features[i]))
        pred = {'feature_maps': out_features}

        if self.conf.compute_uncertainty:
            confidences = []
            for layer, i in zip(self.uncertainty, self.conf.output_scales):
                unc = layer(pre_features[i])
                conf = torch.sigmoid(-unc)
                confidences.append(conf)
            pred['confidences'] = confidences

        return pred

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError
