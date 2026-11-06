"""Differentiable Minnen proxies and checkpoint loading."""

import torch
from torch import nn

from .models import MinnenProxy, pad_rgb


class CompressAIProxy(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, rgb):
        height, width = rgb.shape[-2:]
        result = self.network(pad_rgb(rgb, 64))
        decoded = result['x_hat'][..., :height, :width].clamp(0, 1)
        bits = sum(-likelihood.log2().sum() for likelihood in result['likelihoods'].values())
        return decoded, bits / (rgb.shape[0] * height * width)


def pretrained_proxy(quality):
    try:
        from compressai.zoo import mbt2018
    except ImportError as error:
        raise ValueError('Install requirements-pretrained.txt to use pretrained mbt2018') from error
    return CompressAIProxy(mbt2018(quality=quality, metric='mse', pretrained=True))


def checkpoint_proxy(path, qp, codec_level=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    metadata = checkpoint.get('metadata', {})
    if 'qp' in metadata and metadata['qp'] != qp:
        raise ValueError(f"{path} was trained at QP {metadata['qp']}, expected {qp}")
    if (codec_level is not None and metadata.get('target') == 'bpg'
            and metadata.get('codec_level', codec_level) != codec_level):
        raise ValueError(f"{path} targets BPG level {metadata['codec_level']}, expected level {codec_level}")
    state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    if 'network.g_a.0.weight' in state:
        try:
            from compressai.models import JointAutoregressiveHierarchicalPriors
        except ImportError as error:
            raise ValueError('Install requirements-pretrained.txt to load mbt2018 checkpoints') from error
        network = JointAutoregressiveHierarchicalPriors.from_state_dict(
            {key.removeprefix('network.'): value for key, value in state.items()})
        return CompressAIProxy(network), metadata
    model = MinnenProxy()
    model.load_state_dict(state)
    return model, metadata
