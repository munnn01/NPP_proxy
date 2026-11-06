"""Select pretrained quality levels and measure held-out proxy fidelity."""

import json
import math
from pathlib import Path
from statistics import mean

import torch
from torch.nn import functional as F

from .proxy import checkpoint_proxy, pretrained_proxy
from .tasks import FlickrPatches


def mean_psnr(errors):
    # Use the same finite cap for lossless crops in both codecs.
    return mean(-10 * math.log10(max(error, 1e-12)) for error in errors)


def closest_quality(bpg_psnr, candidates):
    return min(candidates, key=lambda quality: (abs(candidates[quality]['mean_psnr'] - bpg_psnr), quality))


@torch.inference_mode()
def proxy_diagnostics(args, codec, calibrating):
    data = FlickrPatches(args.data, args.crop_size)
    if len(data) < args.samples:
        raise ValueError(f'Need {args.samples} holdout images in {args.data}, found {len(data)}')
    if len(set(args.qps)) != len(args.qps):
        raise ValueError('--qps requires distinct values')
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(data), generator=generator)[:args.samples].tolist()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(args.seed)
        patches = [data[index] for index in indices]
    report = {'schema': 1, 'kind': 'mbt2018-calibration' if calibrating else 'proxy-audit',
              'data': str(args.data.resolve()), 'seed': args.seed, 'crop_size': args.crop_size,
              'codec_level': args.codec_level, 'bpg_encoder': codec.encoder, 'bpg_decoder': codec.decoder,
              'images': [data.paths[index].relative_to(args.data).as_posix() for index in indices],
              'points': {}}

    candidates = {}
    if calibrating:
        for quality in sorted(set(args.qualities)):
            model = pretrained_proxy(quality).to(args.device).eval()
            errors, rates = [], []
            for patch in patches:
                rgb = patch.unsqueeze(0).to(args.device)
                reconstruction, bpp = model(rgb)
                errors.append(F.mse_loss(reconstruction, rgb).item())
                rates.append(bpp.item())
            candidates[quality] = {'mean_psnr': mean_psnr(errors), 'mean_estimated_bpp': mean(rates)}
            del model
        report['candidates'] = candidates

    for qp in args.qps:
        errors, rates, proxy_errors, proxy_rates, alignment = [], [], [], [], []
        if not calibrating:
            model, metadata = checkpoint_proxy(args.proxy_dir / f'qp{qp}.pt', qp)
            model = model.to(args.device).eval()
        for patch in patches:
            decoded, bits = codec.roundtrip(patch, qp)
            errors.append(F.mse_loss(decoded, patch).item())
            rates.append(bits / (patch.shape[-2] * patch.shape[-1]))
            if not calibrating:
                rgb = patch.unsqueeze(0).to(args.device)
                reconstruction, bpp = model(rgb)
                proxy_errors.append(F.mse_loss(reconstruction, rgb).item())
                proxy_rates.append(bpp.item())
                alignment.append(F.mse_loss(reconstruction[0], decoded.to(args.device)).item())
        point = {'bpg_mean_psnr': mean_psnr(errors), 'bpg_mean_bpp': mean(rates)}
        if calibrating:
            point['quality'] = closest_quality(point['bpg_mean_psnr'], candidates)
        else:
            point.update(proxy_mean_psnr=mean_psnr(proxy_errors), proxy_mean_bpp=mean(proxy_rates),
                         proxy_to_bpg_rate_ratio=mean(proxy_rates) / mean(rates),
                         proxy_to_bpg_rgb_mse=mean(alignment), pretrained_quality=metadata.get('pretrained_quality'),
                         proxy_training_codec_level=metadata.get('codec_level'))
            del model
        report['points'][qp] = point
        print(f'QP {qp} {point}', flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def calibrated_quality(path, args):
    report = json.loads(path.read_text(encoding='utf-8'))
    expected = {'schema': 1, 'kind': 'mbt2018-calibration', 'crop_size': args.crop_size,
                'codec_level': args.codec_level}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError('Calibration requires the same crop size and BPG codec level as training')
    holdout, train = (args.calibration_root or Path(report['data'])).resolve(), args.data.resolve()
    if holdout.is_relative_to(train) or train.is_relative_to(holdout):
        raise ValueError('Calibration holdout and Flickr training directories must be separate')
    quality = report['points'].get(str(args.qp), {}).get('quality')
    if quality not in range(1, 9):
        raise ValueError(f'No valid pretrained quality calibrated for QP {args.qp}')
    return quality
