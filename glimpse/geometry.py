"""Geometry helpers for GLIMPSE.

These are the pure, stateless pieces of the GLIMPSE forward model: building the
local sampling "patch" template, the ramp/other Fourier filter used for the
filtered-back-projection (FBP) analogue, and a coordinate-reflection helper used
to keep sampling locations inside the valid image extent.

None of these functions hold learnable state; the learnable versions of these
quantities live on :class:`glimpse.model.GlimpseModel` as ``nn.Parameter``s.
"""

import numpy as np
import torch
from skimage.transform.radon_transform import _get_fourier_filter


def reflect_coords(coords, min_val, max_val):
    """Reflect out-of-range coordinates back into ``[min_val, max_val]``.

    Locations that fall past an edge are mirrored back inside, so a sampling
    patch near the image border still gathers meaningful values instead of
    reading zero-padding. Operates in place on ``coords`` and returns it.
    """
    over = coords[coords > max_val] - max_val
    under = min_val - coords[coords < min_val]

    coords[coords > max_val] = coords[coords > max_val] - 2 * over
    coords[coords < min_val] = coords[coords < min_val] + 2 * under

    return coords


def build_patch_template(patch_shape, patch_rows, patch_cols, image_size):
    """Build the (patch_rows, patch_cols, 2) template of local sample offsets.

    For every pixel we want to reconstruct, GLIMPSE gathers a small set of
    sinogram samples laid out around the pixel coordinate — the "glimpse". This
    function returns the fixed offset pattern (in normalized image units) that
    is added to each pixel coordinate before sampling. When ``learned_patch`` is
    enabled the model registers this template as a learnable parameter and lets
    training reshape the receptive field.

    Parameters
    ----------
    patch_shape : {'round', 'square', 'random'}
        Geometry of the sampling pattern.
    patch_rows, patch_cols : int
        Patch dimensions (both equal to ``w_size``/``patch_size`` in practice).
    image_size : int
        Side length of the reconstructed image, used to normalize offsets.
    """
    n_rows, n_cols = patch_rows, patch_cols

    if patch_shape == 'round':
        radius = n_rows / image_size
        angles = torch.arange(n_cols) * (2 * np.pi / n_cols)
        x = radius * torch.cos(angles) / (2 * n_rows)
        y = radius * torch.sin(angles) / (2 * n_rows)
        x = x[..., None]
        y = y[..., None]
        ring = torch.concat([x, y], dim=1)[None, ...]
        ring = ring.expand(n_rows, -1, -1)
        radial_index = (torch.arange(0, n_rows))[..., None, None]
        patch = radial_index * ring

    elif patch_shape == 'square':
        x = torch.arange(-(n_rows // 2), n_rows // 2 + 1) / image_size
        y = torch.arange(-(n_cols // 2), n_cols // 2 + 1) / image_size
        x, y = torch.meshgrid(x, y, indexing='ij')
        x = x[..., None]
        y = y[..., None]
        patch = torch.concat([x, y], dim=2)[None, ...]

    elif patch_shape == 'random':
        patch = 2 * n_rows * (torch.rand(n_rows, n_cols, 2) - 0.5) / image_size

    else:
        raise ValueError(
            f"Unknown patch_shape {patch_shape!r}; expected 'round', 'square' or 'random'."
        )

    return patch


def init_fourier_filter(filter_name, projection_size_padded):
    """Return the FBP Fourier filter as a float32 tensor.

    Thin wrapper around scikit-image's ``_get_fourier_filter`` so the model can
    initialize its (optionally learnable) filter from a named filter such as
    ``'ramp'`` or ``'shepp-logan'``.
    """
    fourier_filter = _get_fourier_filter(projection_size_padded, filter_name)
    return torch.tensor(fourier_filter, dtype=torch.float32)
