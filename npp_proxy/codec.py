"""Bit-exact payload accounting for the standard BPG command-line codec."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def read_rgb(path):
    with Image.open(path) as image:
        rgb = np.array(image.convert('RGB'), copy=True)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255


def save_rgb(rgb, path):
    pixels = (rgb.detach().cpu().clamp(0, 1) * 255).add(0.5).to(torch.uint8)
    Image.fromarray(pixels.permute(1, 2, 0).numpy()).save(path)


class BPG:
    def __init__(self, encoder='bpgenc', decoder='bpgdec', level=8):
        self.encoder = self.find(encoder)
        self.decoder = self.find(decoder)
        self.level = level

    @staticmethod
    def find(name):
        resolved = shutil.which(str(name))
        if not resolved:
            raise FileNotFoundError(f'{name} unavailable. Install libbpg and set --bpgenc/--bpgdec or PATH.')
        return str(Path(resolved).resolve())

    @staticmethod
    def run(*args):
        try:
            subprocess.run(args, check=True, capture_output=True, text=True, errors='replace', timeout=120)
        except subprocess.CalledProcessError as error:
            detail = error.stderr.strip() or error.stdout.strip()
            raise RuntimeError(f'{Path(args[0]).name} exited {error.returncode}: {detail}') from error

    def encode(self, rgb, qp):
        if not 0 <= qp <= 51:
            raise ValueError(f'BPG QP must be in [0, 51], got {qp}')
        with tempfile.TemporaryDirectory(prefix='npp-bpg-') as temp:
            source = Path(temp) / 'source.png'
            packet = Path(temp) / 'image.bpg'
            save_rgb(rgb, source)
            self.run(self.encoder, '-q', str(qp), '-m', str(self.level), '-o', str(packet), str(source))
            return packet.read_bytes()

    def decode(self, payload):
        if not payload.startswith(b'BPG\xfb'):
            raise ValueError('Invalid BPG signature')
        with tempfile.TemporaryDirectory(prefix='npp-bpg-') as temp:
            packet = Path(temp) / 'image.bpg'
            output = Path(temp) / 'decoded.png'
            packet.write_bytes(payload)
            self.run(self.decoder, '-o', str(output), str(packet))
            return read_rgb(output)

    def roundtrip(self, rgb, qp):
        payload = self.encode(rgb, qp)
        decoded = self.decode(payload)
        if decoded.shape != rgb.shape:
            raise RuntimeError(f'BPG changed the image shape from {rgb.shape} to {decoded.shape}')
        return decoded.to(rgb), 8 * len(payload)
