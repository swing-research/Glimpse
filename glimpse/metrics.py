"""Image-quality metrics and a small parameter-counting helper."""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim


def psnr(reference, estimate):
    """Mean PSNR (dB) over a batch of images, shape ``(n, H, W)``."""
    total = 0.0
    for i in range(np.shape(estimate)[0]):
        total += _psnr(
            reference[i], estimate[i],
            data_range=reference[i].max() - reference[i].min())
    return total / np.shape(estimate)[0]


def ssim(reference, estimate):
    """Mean SSIM over a batch of images, shape ``(n, H, W)``."""
    total = 0.0
    for i in range(np.shape(estimate)[0]):
        total += _ssim(
            reference[i], estimate[i],
            data_range=reference[i].max() - reference[i].min(),
            channel_axis=False)
    return total / np.shape(estimate)[0]


def count_parameters(model):
    """Number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Backwards-compatible uppercase aliases (original utils.py names).
PSNR = psnr
SSIM = ssim
