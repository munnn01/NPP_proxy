"""Paper losses, fixed/adaptive schedules, and resumable optimizer steps."""

import random
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F


PROXY_LAMBDA = {28: 4096, 31: 2048, 34: 1024, 37: 512, 41: 256, 44: 128}
TASK_LAMBDA = {28: 0.5, 31: 1., 34: 2., 37: 4., 41: 8.}
SCHEDULE = {'proxy': (400_000, 320_000), 'fixed': (120_000, 80_000), 'adaptive': (100_000, 60_000)}
BETA = 0.5


def use_value(measured, surrogate):
    """Forward equals measured. The derivative is the derivative of surrogate."""
    return measured.detach() + (surrogate - surrogate.detach())


def fit_proxy_loss(rgb, qp, proxy, codec):
    target = rgb if codec is None else torch.stack([codec.roundtrip(x, qp)[0] for x in rgb])
    reconstructed, bpp = proxy(rgb)
    distortion = F.mse_loss(reconstructed, target)
    loss = bpp + PROXY_LAMBDA[qp] * distortion
    return loss, {'proxy_bpp': bpp.detach().item(), 'target_mse': distortion.detach().item()}


def algorithm1(rgb, qp, npp, proxy, codec, task_loss):
    filtered = npp(rgb, qp)
    surrogate_rgb, surrogate_rate = proxy(filtered)
    if codec is None:
        task_rgb, rate = surrogate_rgb, surrogate_rate
    else:
        decoded, bits = zip(*(codec.roundtrip(image, qp) for image in filtered), strict=True)
        measured_rate = surrogate_rate.new_tensor(sum(bits) / (rgb.shape[0] * rgb.shape[-2] * rgb.shape[-1]))
        task_rgb = use_value(torch.stack(decoded), surrogate_rgb)
        rate = use_value(measured_rate, surrogate_rate)
    task = task_loss(task_rgb)
    pre_mse = F.mse_loss(rgb, filtered)
    loss = rate + TASK_LAMBDA[qp] * task + BETA * pre_mse
    return loss, {'bpp': rate.detach().item(), 'proxy_bpp': surrogate_rate.detach().item(),
                  'task': task.detach().item(), 'pre_mse': pre_mse.detach().item()}


def load_state(path):
    contents = torch.load(path, map_location='cpu', weights_only=True)
    return contents.get('model', contents.get('state_dict', contents))


def save_checkpoint(path, model, optimizer, step, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'step': step, 'metadata': metadata}, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def optimize(args, model, stage, dataset, update, metadata):
    max_steps, decay = SCHEDULE[stage]
    max_steps = max_steps if args.steps is None else args.steps
    decay = decay if args.decay is None else args.decay
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    metadata = {**metadata, 'stage': stage, 'seed': args.seed, 'batch_size': args.batch_size,
                'decay': decay, 'data': str(Path(args.data).resolve())}
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=True)
        if state['metadata'] != metadata:
            raise ValueError('Resume requires matching training data and configuration')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start = state['step']
    if max_steps <= start:
        raise ValueError(f'--steps ({max_steps}) must exceed completed steps ({start})')

    for step in range(start, max_steps):
        rng = random.Random(args.seed + step)
        torch.manual_seed(args.seed + step)
        batch = [dataset[i] for i in rng.choices(range(len(dataset)), k=args.batch_size)]
        optimizer.param_groups[0]['lr'] = 1e-5 if step >= decay else 1e-4
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = update(batch, rng)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite loss at step {step + 1}: {metrics}')
        loss.backward()
        optimizer.step()
        completed = step + 1
        if completed == 1 or completed % args.log_every == 0 or completed == max_steps:
            values = ' '.join(f'{key}={value:.6g}' for key, value in metrics.items())
            print(f'{stage} {completed}/{max_steps} loss={loss.item():.6g} {values}', flush=True)
        if completed % args.save_every == 0 or completed == max_steps:
            save_checkpoint(args.output, model, optimizer, completed, metadata)
