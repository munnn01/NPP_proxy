"""Command-line entry points for training, evaluation and single-image coding."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

from .codec import BPG, read_rgb, save_rgb
from .metrics import bd_rate
from .models import NPP, MinnenProxy
from .tasks import CLASSIFIERS, DETECTORS, FlickrPatches, Task
from .training import PROXY_LAMBDA, TASK_LAMBDA, algorithm1, fit_proxy_loss, load_state, optimize


def codec_for(args):
    return BPG(args.bpgenc, args.bpgdec, args.codec_level)


def load_npp(path, device):
    state = load_state(path)
    model = NPP(adaptive='Scaling_Net1.layer.0.weight' in state).to(device)
    model.load_state_dict(state)
    return model.eval()


def train_proxy(args):
    if args.target == 'bpg' and not (args.init or args.resume or args.from_scratch):
        raise ValueError('BPG fine-tuning requires --init, --resume, or explicit --from-scratch')
    codec = codec_for(args) if args.target == 'bpg' else None
    data = FlickrPatches(args.data, args.crop_size)
    if not len(data):
        raise ValueError(f'No proxy training images found in {args.data}')
    proxy = MinnenProxy().to(args.device).train()
    if args.init:
        proxy.load_state_dict(load_state(args.init))

    def update(batch, _rng):
        return fit_proxy_loss(torch.stack(batch).to(args.device), args.qp, proxy, codec)

    optimize(args, proxy, 'proxy', data, update,
             {'qp': args.qp, 'target': args.target, 'crop_size': args.crop_size, 'codec_level': args.codec_level})


def train_npp(args):
    adaptive = args.stage == 'adaptive'
    if adaptive and not (args.fixed or args.resume):
        raise ValueError('Adaptive training requires --fixed or --resume')
    if not adaptive and args.fixed:
        raise ValueError('--fixed initializes the adaptive stage')
    npp = NPP(adaptive).to(args.device).train()
    if args.fixed:
        npp.load_fixed(load_state(args.fixed))
    codec = None if args.proxy_only else codec_for(args)
    task = Task(args.task, None, args.device, args.task_weights)
    data = task.dataset(args.data, args.annotations, train=True)
    if not len(data):
        raise ValueError(f'No task training images found in {args.data}')
    qps = tuple(TASK_LAMBDA) if adaptive else (args.qp,)
    proxies = {}
    for qp in qps:
        proxy_path = args.proxy_dir / f'qp{qp}.pt'
        model = MinnenProxy().to(args.device)
        model.load_state_dict(load_state(proxy_path))
        # Training mode adds quantization noise and preserves input gradients. Weights stay frozen.
        proxies[qp] = model.train().requires_grad_(False)

    def update(batch, rng):
        qp = rng.choice(qps)
        if args.task == 'classification':
            images, labels = zip(*batch, strict=True)
            return algorithm1(torch.stack(images).to(args.device), qp, npp, proxies[qp], codec,
                              lambda decoded: task.loss(decoded, labels))
        samples = [algorithm1(image.unsqueeze(0).to(args.device), qp, npp, proxies[qp], codec,
                              lambda decoded: task.loss(decoded, target)) for image, target, _ in batch]
        metrics = {key: sum(values[key] for _, values in samples) / len(samples) for key in samples[0][1]}
        return torch.stack([loss for loss, _ in samples]).mean(), metrics

    optimize(args, npp, args.stage, data, update,
             {'task': args.task, 'backbone': task.backbone, 'task_weights': task.weight_id,
              'proxy_dir': str(args.proxy_dir.resolve()), 'qps': list(qps), 'proxy_only': args.proxy_only,
              'codec_level': args.codec_level,
              'annotations': str(args.annotations.resolve()) if args.annotations else None})


@torch.no_grad()
def evaluate(args):
    if len(set(args.qps)) != len(args.qps):
        raise ValueError('--qps requires distinct values')
    codec = codec_for(args)
    npp = load_npp(args.npp, args.device)
    task = Task(args.task, args.backbone, args.device, args.task_weights)
    data = task.dataset(args.data, args.annotations)
    count = min(len(data), args.limit) if args.limit is not None else len(data)
    if not count:
        raise ValueError('Evaluation requires at least one image')
    rows = []
    for qp in args.qps:
        rates, bit_totals, results = defaultdict(list), defaultdict(int), defaultdict(list)
        total_pixels = 0
        ids = []
        for index in range(count):
            example = data[index]
            source = example[0].to(args.device)
            filtered = npp(source.unsqueeze(0), qp)[0]
            pixel_count = source.shape[-2] * source.shape[-1]
            total_pixels += pixel_count
            if args.task == 'detection':
                ids.append(example[2])
            for variant, rgb in (('bpg', source), ('npp+bpg', filtered)):
                decoded, bits = codec.roundtrip(rgb, qp)
                rates[variant].append(bits / pixel_count)
                bit_totals[variant] += bits
                prediction = task.predict(decoded, example)
                if args.task == 'classification':
                    results[variant].append(prediction)
                else:
                    results[variant].extend(prediction)
        for variant, values in rates.items():
            if args.task == 'detection':
                metric = data.evaluate(results[variant], ids)
            else:
                metric = sum(results[variant]) / count * 100
            row = {'variant': variant, 'qp': qp, 'images': count, 'mean_bpp': sum(values) / count,
                   'pooled_bpp': bit_totals[variant] / total_pixels, 'metric_percent': metric}
            rows.append(row)
            print(row, flush=True)
    curves = {name: [(r['mean_bpp'], r['metric_percent']) for r in rows if r['variant'] == name]
              for name in ('bpg', 'npp+bpg')}
    report = {'task': args.task, 'backbone': task.backbone, 'task_weights': task.weight_id,
              'metric': 'COCO AP@[.50:.95]' if args.task == 'detection' else 'ImageNet top-1',
              'checkpoint': str(args.npp.resolve()), 'bpg_encoder': codec.encoder, 'bpg_decoder': codec.decoder,
              'codec_level': args.codec_level, 'bd_rate': bd_rate(curves['bpg'], curves['npp+bpg']), 'rows': rows}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (args.output / 'rate_task.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report['bd_rate']), flush=True)


@torch.no_grad()
def encode_image(args):
    codec = codec_for(args)
    rgb = read_rgb(args.input).to(args.device)
    if args.npp:
        rgb = load_npp(args.npp, args.device)(rgb.unsqueeze(0), args.qp)[0]
    payload = codec.encode(rgb, args.qp)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    if args.decoded:
        args.decoded.parent.mkdir(parents=True, exist_ok=True)
        save_rgb(codec.decode(payload), args.decoded)
    print(json.dumps({'qp': args.qp, 'bytes': len(payload), 'bpp': 8 * len(payload) / (rgb.shape[-2] * rgb.shape[-1])}))


def positive(text):
    number = int(text)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest='command', required=True)

    def command(name, action):
        sub = commands.add_parser(name)
        sub.set_defaults(action=action)
        sub.add_argument('--device', default='auto')
        sub.add_argument('--threads', type=positive, default=4)
        sub.add_argument('--bpgenc', default='bpgenc')
        sub.add_argument('--bpgdec', default='bpgdec')
        sub.add_argument('--codec-level', type=int, choices=range(1, 10), default=8)
        sub.add_argument('--output', required=True, type=Path)
        return sub

    def training(sub):
        sub.add_argument('--data', required=True, type=Path)
        sub.add_argument('--steps', type=positive)
        sub.add_argument('--decay', type=int)
        sub.add_argument('--batch-size', type=positive, default=1)
        sub.add_argument('--seed', type=int, default=2024)
        sub.add_argument('--resume', type=Path)
        sub.add_argument('--log-every', type=positive, default=100)
        sub.add_argument('--save-every', type=positive, default=1000)

    def task_flags(sub):
        sub.add_argument('--task', required=True, choices=['classification', 'detection'])
        sub.add_argument('--annotations', type=Path)
        sub.add_argument('--task-weights', type=Path)

    proxy = command('train-proxy', train_proxy)
    training(proxy)
    proxy.add_argument('--qp', required=True, type=int, choices=PROXY_LAMBDA)
    proxy.add_argument('--target', choices=['source', 'bpg'], default='bpg')
    proxy.add_argument('--crop-size', type=positive, default=256)
    initial = proxy.add_mutually_exclusive_group()
    initial.add_argument('--init', type=Path)
    initial.add_argument('--from-scratch', action='store_true')

    npp = command('train-npp', train_npp)
    training(npp)
    task_flags(npp)
    npp.add_argument('--stage', required=True, choices=['fixed', 'adaptive'])
    npp.add_argument('--qp', type=int, choices=TASK_LAMBDA, default=34)
    npp.add_argument('--proxy-dir', required=True, type=Path)
    npp.add_argument('--fixed', type=Path)
    npp.add_argument('--proxy-only', action='store_true')

    evaluation = command('evaluate', evaluate)
    task_flags(evaluation)
    evaluation.add_argument('--data', required=True, type=Path)
    evaluation.add_argument('--npp', required=True, type=Path)
    evaluation.add_argument('--backbone', choices=[*CLASSIFIERS, *DETECTORS])
    evaluation.add_argument('--qps', nargs='+', type=int, choices=range(52), default=list(TASK_LAMBDA))
    evaluation.add_argument('--limit', type=positive)

    single = command('encode', encode_image)
    single.add_argument('--input', required=True, type=Path)
    single.add_argument('--npp', type=Path)
    single.add_argument('--qp', required=True, type=int, choices=range(52))
    single.add_argument('--decoded', type=Path)
    return root


def main():
    root = parser()
    args = root.parse_args()
    torch.set_num_threads(args.threads)
    if hasattr(args, 'seed'):
        torch.manual_seed(args.seed)
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        args.action(args)
    except (ValueError, FileNotFoundError) as error:
        root.error(str(error))


if __name__ == '__main__':
    main()
