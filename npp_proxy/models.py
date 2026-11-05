"""NPP and Minnen proxy adapted from XingtongGe/PreprocessingICM (Apache-2.0)."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def pad_rgb(x, multiple):
    height, width = x.shape[-2:]
    padded_height = max(32, math.ceil(height / multiple) * multiple)
    padded_width = max(32, math.ceil(width / multiple) * multiple)
    return F.pad(x, (0, padded_width - width, 0, padded_height - height), mode='replicate')


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels), nn.ReLU(),
    )


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Conv_BN_ReLU_2 = conv_block(in_channels, out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
        )

    def forward(self, x):
        skip = self.Conv_BN_ReLU_2(x)
        return skip, self.downsample(skip)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Conv_BN_ReLU_2 = conv_block(in_channels, 2 * out_channels)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(2 * out_channels, out_channels, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
        )

    def forward(self, x, skip):
        return torch.cat((self.upsample(self.Conv_BN_ReLU_2(x)), skip), dim=1)


class QPScale(nn.Module):
    def __init__(self, hidden, channels):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, channels), nn.ReLU())
        nn.init.zeros_(self.layer[2].weight)
        nn.init.ones_(self.layer[2].bias)

    def forward(self, features, condition):
        return features * self.layer(condition)[:, :, None, None]


class UNet(nn.Module):
    def __init__(self, adaptive):
        super().__init__()
        self.d1 = Downsample(3, 32)
        self.d2 = Downsample(32, 64)
        self.d3 = Downsample(64, 128)
        self.d4 = Downsample(128, 256)
        self.u1 = Upsample(256, 256)
        self.u2 = Upsample(512, 128)
        self.u3 = Upsample(256, 64)
        self.u4 = Upsample(128, 32)
        self.o = conv_block(64, 32)
        self.o.extend((nn.Conv2d(32, 3, 1), nn.ReLU()))
        self.adaptive = adaptive
        if adaptive:
            self.Scaling_Net1 = QPScale(32, 32)
            self.Scaling_Net2 = QPScale(32, 64)
            self.Scaling_Net3 = QPScale(64, 128)
            self.Scaling_Net4 = QPScale(64, 256)
            self.Scaling_Net5 = QPScale(64, 128)
            self.Scaling_Net6 = QPScale(32, 64)

    def forward(self, x, qp):
        skip1, x = self.d1(x)
        skip2, x = self.d2(self.Scaling_Net1(x, qp) if self.adaptive else x)
        skip3, x = self.d3(self.Scaling_Net2(x, qp) if self.adaptive else x)
        skip4, x = self.d4(self.Scaling_Net3(x, qp) if self.adaptive else x)
        x = self.u2(self.u1(x, skip4), skip3)
        x = self.u3(self.Scaling_Net4(x, qp) if self.adaptive else x, skip2)
        x = self.u4(self.Scaling_Net5(x, qp) if self.adaptive else x, skip1)
        return self.o(self.Scaling_Net6(x, qp) if self.adaptive else x)


class NPP(nn.Module):
    """Two branches summed in RGB space. QP gates initialize to identity."""

    def __init__(self, adaptive=False):
        super().__init__()
        self.adaptive = adaptive
        self.conv1 = nn.Conv2d(3, 3, 1)
        self.conv2 = nn.Conv2d(3, 3, 1)
        self.UNet = UNet(adaptive)
        if adaptive:
            self.Scaling_Net1 = QPScale(16, 3)
            self.Scaling_Net2 = QPScale(16, 3)

    def load_fixed(self, state):
        expected = {name for name in self.state_dict() if 'Scaling_Net' not in name}
        if not self.adaptive or state.keys() != expected:
            raise ValueError('Adaptive NPP requires a complete fixed NPP state')
        self.load_state_dict(state, strict=False)

    def forward(self, x, qp):
        height, width = x.shape[-2:]
        condition = torch.as_tensor(qp, device=x.device, dtype=x.dtype).reshape(-1, 1) / 51
        if condition.shape[0] not in (1, x.shape[0]):
            raise ValueError('Pass one QP for the batch or one QP per image')
        shallow = self.conv1(x)
        shallow = F.relu(self.Scaling_Net1(shallow, condition) if self.adaptive else shallow)
        shallow = self.conv2(shallow)
        shallow = F.relu(self.Scaling_Net2(shallow, condition) if self.adaptive else shallow)
        deep = self.UNet(pad_rgb(x, 16), condition)[..., :height, :width]
        return (shallow + deep).clamp(0, 1)


class LowerBound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, minimum):
        ctx.save_for_backward(values)
        ctx.minimum = minimum
        return values.clamp_min(minimum)

    @staticmethod
    def backward(ctx, gradient):
        values, = ctx.saved_tensors
        return gradient * ((values >= ctx.minimum) | (gradient < 0)), None


class GDN(nn.Module):
    def __init__(self, channels, inverse=False):
        super().__init__()
        self.inverse = inverse
        self.offset = 2 ** -18
        self.beta = nn.Parameter(torch.full((channels,), math.sqrt(1 + self.offset ** 2)))
        self.gamma = nn.Parameter(torch.sqrt(0.1 * torch.eye(channels) + self.offset ** 2))

    def forward(self, x):
        beta = LowerBound.apply(self.beta, math.sqrt(1e-6 + self.offset ** 2)).square() - self.offset ** 2
        gamma = LowerBound.apply(self.gamma, self.offset).square() - self.offset ** 2
        norm = F.conv2d(x.square(), gamma[:, :, None, None], beta).sqrt()
        return x * norm if self.inverse else x / norm


def initialized(layer, gain=math.sqrt(2)):
    nn.init.xavier_normal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.01)
    return layer


class Analysis(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = initialized(nn.Conv2d(3, channels, 5, 2, 2), math.sqrt(2 * (3 + channels)) / 6)
        self.conv2 = initialized(nn.Conv2d(channels, channels, 5, 2, 2))
        self.conv3 = initialized(nn.Conv2d(channels, channels, 5, 2, 2))
        self.conv4 = initialized(nn.Conv2d(channels, channels, 5, 2, 2))
        self.gdn1, self.gdn2, self.gdn3 = (GDN(channels) for _ in range(3))

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        return self.conv4(self.gdn3(self.conv3(x)))


class Synthesis(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.deconv1 = initialized(nn.ConvTranspose2d(channels, channels, 5, 2, 2, output_padding=1))
        self.deconv2 = initialized(nn.ConvTranspose2d(channels, channels, 5, 2, 2, output_padding=1))
        self.deconv3 = initialized(nn.ConvTranspose2d(channels, channels, 5, 2, 2, output_padding=1))
        self.deconv4 = initialized(nn.ConvTranspose2d(channels, 3, 5, 2, 2, output_padding=1),
                                   math.sqrt((channels + 3) / channels))
        self.igdn1, self.igdn2, self.igdn3 = (GDN(channels, inverse=True) for _ in range(3))

    def forward(self, x):
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        return self.deconv4(self.igdn3(self.deconv3(x)))


class HyperAnalysis(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = initialized(nn.Conv2d(c, c, 3, padding=1))
        self.conv2 = initialized(nn.Conv2d(c, c, 5, 2, 2))
        self.conv3 = initialized(nn.Conv2d(c, c, 5, 2, 2))

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        return self.conv3(F.leaky_relu(self.conv2(x), 0.2))


class HyperSynthesis(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.deconv1 = initialized(nn.ConvTranspose2d(c, c, 5, 2, 2, output_padding=1))
        self.deconv2 = initialized(nn.ConvTranspose2d(c, c + c // 2, 5, 2, 2, output_padding=1), math.sqrt(2.5))
        self.deconv3 = initialized(nn.ConvTranspose2d(c + c // 2, 2 * c, 3, padding=1), math.sqrt(7 / 3))

    def forward(self, x):
        x = F.leaky_relu(self.deconv1(x), 0.2)
        return self.deconv3(F.leaky_relu(self.deconv2(x), 0.2))


class CausalConv(nn.Conv2d):
    def __init__(self, channels):
        super().__init__(channels, 2 * channels, 5, padding=2)
        initialized(self, math.sqrt(3))
        mask = torch.ones_like(self.weight)
        mask[:, :, 2, 2:] = 0
        mask[:, :, 3:, :] = 0
        self.register_buffer('mask', mask)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias, padding=2)


class Context(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = CausalConv(c)

    def forward(self, x):
        return self.conv1(x)


class EntropyParameters(nn.Module):
    def __init__(self, c):
        super().__init__()
        a, b = 10 * c // 3, 8 * c // 3
        self.conv1 = initialized(nn.Conv2d(4 * c, a, 1), math.sqrt((4 * c + a) / (4 * c)))
        self.conv2 = initialized(nn.Conv2d(a, b, 1), math.sqrt((a + b) / a))
        self.conv3 = initialized(nn.Conv2d(b, 2 * c, 1), math.sqrt((b + 2 * c) / b))

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        return self.conv3(F.leaky_relu(self.conv2(x), 0.2))


class CDFLayer(nn.Module):
    def __init__(self, c, final=False):
        super().__init__()
        self.h = nn.Parameter(torch.randn(1, c, 1, 1) * 0.01)
        self.b = nn.Parameter(torch.randn(1, c, 1, 1) * 0.01)
        self.a = None if final else nn.Parameter(torch.randn(1, c, 1, 1) * 0.01)

    def forward(self, x):
        x = x * F.softplus(self.h) + self.b
        return x.sigmoid() if self.a is None else x + x.tanh() * self.a.tanh()


class HyperCDF(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.f1, self.f2, self.f3 = (CDFLayer(c) for _ in range(3))
        self.f4 = CDFLayer(c, final=True)

    def forward(self, x):
        return self.f4(self.f3(self.f2(self.f1(x))))


class MinnenProxy(nn.Module):
    """192-channel joint autoregressive/hyperprior model from the public implementation."""

    def __init__(self, channels=192):
        super().__init__()
        self.Encoder = Analysis(channels)
        self.Decoder = Synthesis(channels)
        self.priorEncoder = HyperAnalysis(channels)
        self.priorDecoder = HyperSynthesis(channels)
        self.bitEstimator_z = HyperCDF(channels)
        self.context_prediction = Context(channels)
        self.entropy_parameters = EntropyParameters(channels)

    def quantize(self, latent):
        if self.training:
            return latent + torch.empty_like(latent).uniform_(-0.5, 0.5)
        return latent.round()

    def forward(self, rgb):
        height, width = rgb.shape[-2:]
        y = self.Encoder(pad_rgb(rgb, 64))
        z_hat = self.quantize(self.priorEncoder(y))
        y_hat = self.quantize(y)
        hyper = self.priorDecoder(z_hat)
        context = self.context_prediction(y_hat)
        mean, scale = self.entropy_parameters(torch.cat((hyper, context), dim=1)).chunk(2, dim=1)
        scale = scale.square().clamp(1e-5, 1e5)
        normalizer = scale * math.sqrt(2)
        y_probability = 0.5 * (((y_hat + 0.5 - mean) / normalizer).erf()
                               - ((y_hat - 0.5 - mean) / normalizer).erf())
        z_probability = self.bitEstimator_z(z_hat + 0.5) - self.bitEstimator_z(z_hat - 0.5)
        bits = sum(-(p + 1e-5).log2().clamp_max(0).sum() for p in (y_probability, z_probability))
        recon = self.Decoder(y_hat)[..., :height, :width].clamp(0, 1)
        return recon, bits / (rgb.shape[0] * height * width)
