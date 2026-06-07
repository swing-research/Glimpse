"""Coordinate grids, full-image reconstruction, and the FBP baseline.

GLIMPSE predicts one pixel at a time, so reconstructing a whole image means
evaluating the model on every pixel coordinate. These helpers build the
coordinate grid, tile it through the model in chunks, and provide the classical
filtered-back-projection (FBP) baseline for comparison.
"""

import numpy as np
import torch
from skimage.transform import iradon


def make_coordinate_grid(side_length):
    """Return all pixel coordinates of a ``side_length`` square image.

    Coordinates are normalized to ``[-0.5, 0.5]`` (the convention the model
    expects), shape ``(side_length**2, 2)``.
    """
    coords = np.stack(np.mgrid[:side_length, :side_length], axis=-1).astype(np.float32)
    coords /= (side_length - 1)
    coords -= 0.5
    return torch.Tensor(coords).reshape(-1, 2)


def reconstruct_image(sinogram, coords, channels, model, chunk_size=512):
    """Reconstruct full images by evaluating ``model`` over all ``coords``.

    Parameters
    ----------
    sinogram : Tensor, shape (batch, n_detector_bins, n_angles)
    coords : Tensor, shape (batch, n_pixels, 2)
        Typically the output of :func:`make_coordinate_grid`, expanded to batch.
    channels : int
        Output channels per pixel (1 for grayscale CT).
    model : GlimpseModel
    chunk_size : int
        Number of pixel coordinates evaluated per forward pass (memory knob).

    Returns
    -------
    np.ndarray, shape (batch, n_pixels, channels).
    """
    n_pixels = np.shape(coords)[1]
    out = np.zeros([np.shape(coords)[0], n_pixels, channels])
    with torch.no_grad():
        for i in range(int(np.ceil(n_pixels / chunk_size))):
            batch_coords = coords[:, i * chunk_size: (i + 1) * chunk_size]
            pred = model(batch_coords, sinogram).detach().cpu().numpy()
            out[:, i * chunk_size: (i + 1) * chunk_size] = pred
    return out


def fbp_batch(sinograms, theta):
    """Classical filtered-back-projection baseline for a batch of sinograms."""
    fbps = [iradon(sinograms[i], theta=theta, circle=False) for i in range(sinograms.shape[0])]
    return np.array(fbps)
