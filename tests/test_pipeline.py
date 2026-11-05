import shutil
import unittest

import torch
from torch.nn import functional as F

from npp_proxy.codec import BPG
from npp_proxy.metrics import bd_rate
from npp_proxy.models import NPP, MinnenProxy
from npp_proxy.training import algorithm1, fit_proxy_loss, use_value


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(32)
        torch.set_num_threads(2)

    def test_real_values_and_proxy_gradient(self):
        source = torch.tensor([1., 3.], requires_grad=True)
        measured = torch.tensor([4., 7.], requires_grad=True)
        result = use_value(measured, source.square())
        self.assertTrue(torch.equal(result, measured))
        result.square().sum().backward()
        torch.testing.assert_close(source.grad, 4 * measured.detach() * source.detach())
        self.assertIsNone(measured.grad)

    def test_fixed_to_adaptive_preserves_initial_function(self):
        fixed = NPP(adaptive=False).eval()
        adaptive = NPP(adaptive=True).eval()
        adaptive.load_fixed(fixed.state_dict())
        x = torch.rand(2, 3, 45, 67)
        with torch.no_grad():
            expected = fixed(x, 34)
            actual = adaptive(x, torch.tensor([28, 41]))
        self.assertEqual(x.shape, actual.shape)
        torch.testing.assert_close(expected, actual)
        self.assertEqual(sum(p.numel() for p in fixed.parameters()), 9_422_235)
        self.assertEqual(sum(p.numel() for p in adaptive.parameters()), 9_461_537)

    def test_minnen_padding_and_gradient(self):
        proxy = MinnenProxy().train().requires_grad_(False)
        x = torch.rand(1, 3, 55, 67, requires_grad=True)
        reconstructed, bpp = proxy(x)
        self.assertEqual(x.shape, reconstructed.shape)
        self.assertGreater(bpp.item(), 0)
        for output in (bpp, reconstructed.mean()):
            gradient, = torch.autograd.grad(output, x, retain_graph=True)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_bd_rate_ratio_and_disjoint_curves(self):
        anchor = [(0.1, 30), (0.2, 40), (0.4, 50), (0.8, 60)]
        candidate = [(rate * 0.8, quality) for rate, quality in anchor]
        result = bd_rate(anchor, candidate)
        self.assertAlmostEqual(result['percent'], -20)
        self.assertEqual(result['interval'], [30, 60])
        self.assertIsNone(bd_rate(anchor, [(r, q + 100) for r, q in candidate])['percent'])

    @unittest.skipUnless(shutil.which('bpgenc') and shutil.which('bpgdec'), 'BPG tools not on PATH')
    def test_bpg_and_three_training_stages(self):
        codec = BPG()
        x = torch.rand(1, 3, 63, 65)
        proxy = MinnenProxy()
        optimizer = torch.optim.Adam(proxy.parameters(), lr=1e-4)
        proxy_loss, _ = fit_proxy_loss(x, 34, proxy, codec)
        proxy_loss.backward()
        optimizer.step()
        proxy.requires_grad_(False).zero_grad(set_to_none=True)
        fixed = NPP(adaptive=False)
        adaptive = NPP(adaptive=True)
        for npp, qp in ((fixed, 34), (adaptive, 41)):
            if npp is adaptive:
                npp.load_fixed(fixed.state_dict())
            filtered = npp(x, qp)
            decoded, bits = codec.roundtrip(filtered[0], qp)
            measured = codec.encode(filtered[0], qp)
            self.assertEqual(bits, 8 * len(measured))
            self.assertEqual(decoded.shape, x.shape[1:])
            task = torch.nn.Conv2d(3, 2, 1).requires_grad_(False)
            optimizer = torch.optim.Adam(npp.parameters(), lr=1e-4)
            loss, values = algorithm1(x, qp, npp, proxy, codec, lambda rgb: task(rgb).square().mean())
            self.assertAlmostEqual(values['bpp'], bits / (63 * 65), places=6)
            self.assertAlmostEqual(values['task'], task(decoded.unsqueeze(0)).square().mean().item(), places=6)
            self.assertAlmostEqual(values['pre_mse'], F.mse_loss(x, filtered).item(), places=6)
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in npp.parameters()))
            self.assertTrue(all(p.grad is None for p in list(proxy.parameters()) + list(task.parameters())))
            optimizer.step()


if __name__ == '__main__':
    unittest.main()
