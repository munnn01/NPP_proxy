import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from npp_proxy.codec import save_rgb
from npp_proxy.models import NPP
from npp_proxy.proxy import CompressAIProxy
from npp_proxy.proxy_diagnostics import calibrated_quality, proxy_diagnostics
from npp_proxy.training import EpochShuffle, fixed_sample_offset, optimize


class ToyMinnen(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, rgb):
        probability = 0.25 + 0.5 * rgb.mean(dim=(2, 3))
        return {'x_hat': self.scale * rgb, 'likelihoods': {'y': probability, 'z': probability.square()}}


class FidelityTests(unittest.TestCase):
    def test_identity_initialization_preserves_rgb8_and_deep_gradient(self):
        model = NPP().eval()
        model.initialize_identity()
        rgb = torch.randint(0, 256, (1, 3, 35, 47)).float() / 255
        filtered = model(rgb, 34)
        torch.testing.assert_close((filtered * 255).round(), (rgb * 255).round(), atol=0, rtol=0)
        filtered.mean().backward()
        self.assertGreater(model.UNet.o[-2].weight.grad.abs().sum().item(), 0)
        adaptive = NPP(adaptive=True).eval()
        adaptive.load_fixed(model.state_dict())
        torch.testing.assert_close(adaptive(rgb, 41), filtered)

    def test_pretrained_adapter_preserves_input_shape_and_gradients(self):
        rgb = torch.full((2, 3, 31, 65), 0.5, requires_grad=True)
        decoded, bpp = CompressAIProxy(ToyMinnen())(rgb)
        torch.testing.assert_close(decoded, rgb)
        self.assertAlmostEqual(bpp.item(), 18 / (2 * 31 * 65))
        for scalar in (decoded.mean(), bpp):
            gradient, = torch.autograd.grad(scalar, rgb, retain_graph=True)
            self.assertTrue(gradient.isfinite().all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_epoch_resume_and_adaptive_stream(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            dataset = list(range(17))
            args = SimpleNamespace(data=root, seed=2024, batch_size=2, steps=3, decay=3,
                                   resume=None, output=root / 'fixed.pt', save_every=1, log_every=100)
            metadata = {'task': 'classification', 'annotations': None}
            visited = []

            def train(model):
                def update(batch, rng):
                    visited.extend(batch)
                    return (model(torch.rand(1)) - rng.random()).square().sum(), {}
                optimize(args, model, 'fixed', dataset, update, metadata)

            torch.manual_seed(1)
            model = nn.Linear(1, 1)
            train(model)
            checkpoint = torch.load(args.output, weights_only=True)
            offset = fixed_sample_offset(checkpoint, root, args.seed, 'classification', None)
            sampler = EpochShuffle(len(dataset), args.seed)
            self.assertEqual(offset, 6)
            self.assertEqual(visited, sampler.batch(0, offset))
            self.assertTrue(set(visited).isdisjoint(sampler.batch(offset, 6)))
            self.assertEqual(len(set(sampler.batch(0, len(dataset)))), len(dataset))
            self.assertEqual(sampler.batch(16, 3), sampler.batch(16, 1) + sampler.batch(17, 2))

            args.steps, args.resume = 6, args.output
            train(model)
            resumed = model.state_dict()
            self.assertEqual(visited, sampler.batch(0, 12))
            args.resume, args.output = None, root / 'continuous.pt'
            torch.manual_seed(1)
            continuous = nn.Linear(1, 1)
            train(continuous)
            for name, value in continuous.state_dict().items():
                torch.testing.assert_close(resumed[name], value, atol=0, rtol=0)

    def test_calibration_uses_separate_holdout_and_matching_psnr(self):
        class ToyBPG:
            encoder = 'toy-encoder'
            decoder = 'toy-decoder'

            def roundtrip(self, rgb, _qp):
                return rgb * 0.8, 128

        with TemporaryDirectory() as folder:
            root = Path(folder)
            holdout = root / 'holdout'
            holdout.mkdir()
            save_rgb(torch.rand(3, 31, 35), holdout / 'image.png')
            args = SimpleNamespace(data=holdout, crop_size=32, samples=1, seed=7, device='cpu', codec_level=8,
                                   qualities=[1, 2], qps=[34], output=root / 'calibration.json')
            with patch('npp_proxy.proxy_diagnostics.pretrained_proxy',
                       side_effect=lambda quality: CompressAIProxy(ToyMinnen({1: 0.5, 2: 0.8}[quality]))):
                proxy_diagnostics(args, ToyBPG(), calibrating=True)
            report = json.loads(args.output.read_text())
            self.assertEqual(report['images'], ['image.png'])
            self.assertEqual(report['points']['34']['quality'], 2)
            args.qp, args.calibration_root = 34, None
            args.data = root / 'train'
            self.assertEqual(calibrated_quality(args.output, args), 2)
            args.data = root
            with self.assertRaisesRegex(ValueError, 'must be separate'):
                calibrated_quality(args.output, args)
