"""Dataset loading for GLIMPSE.

:class:`RawImageCTDataset` wraps **raw images** (``.npy`` arrays, or ``.jpg`` /
``.png`` files) — exactly what the published SwitchDrive datasets contain
(``train``/``test`` are raw ``.npy`` CT images, ``ood`` are brain ``.jpg``).
Sinograms are produced on the fly: by the ODL operator on the GPU (default
``geometry='odl'``) or by skimage ``radon`` (``geometry='skimage'``).

For the ``'glimpse'`` network the loader yields ``(image_flattened_(H*W, 1),
sinogram)``; for ``'odl'`` it yields ``(image_flattened, image_2d)`` and the
engine projects the image with the ODL operator.
"""

import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import iradon, radon

# Files we treat as data; everything else (e.g. macOS ``.DS_Store`` /
# ``._*`` junk) is ignored when listing a dataset directory.
_DATA_EXTENSIONS = {'.npy', '.jpg', '.jpeg', '.png'}


def _list_data_files(directory):
    """Sorted data files in ``directory``, skipping dotfiles and junk."""
    names = []
    for name in os.listdir(directory):
        if name.startswith('.'):
            continue
        if os.path.splitext(name)[1].lower() in _DATA_EXTENSIONS:
            names.append(name)
    return sorted(names)


def _add_measurement_noise(sinogram, noise_snr, rng):
    """Add Gaussian measurement noise at the given SNR (dB) to a sinogram."""
    sigma = 10 ** (-noise_snr / 20.0) * np.sqrt(
        np.mean(np.sum(np.square(np.reshape(sinogram, (1, -1))), -1)))
    noise = rng.normal(loc=0, scale=sigma, size=np.shape(sinogram))
    noise /= np.sqrt(np.prod(np.shape(sinogram)))
    return sinogram + noise


class RawImageCTDataset(torch.utils.data.Dataset):
    """Render sinograms on the fly from raw images.

    Parameters
    ----------
    directory : str
        Folder of raw images (``.npy`` arrays, or ``.jpg``/``.png`` when
        ``image_kind='image_file'``).
    image_size : int
        Images are resized to ``image_size`` x ``image_size`` before projection.
    true_angles_deg : np.ndarray
        Projection angles used to generate the sinogram (the real geometry).
    fbp_angles_deg : np.ndarray
        Angles used for the FBP baseline (only when ``return_fbp=True``); may
        differ from ``true_angles_deg`` to model miscalibration.
    noise_snr : float
        Measurement-noise level in dB.
    return_fbp : bool
        If True yield ``(image, fbp)`` (for the U-Net baseline); otherwise yield
        ``(image_flattened, sinogram)`` for GLIMPSE.
    image_kind : {'array', 'image_file'}
        ``'array'`` loads ``.npy``; ``'image_file'`` reads ``.jpg``/``.png`` and
        normalizes to ``[0, 1]`` (used for the OOD brain set).
    """

    def __init__(self, directory, image_size, true_angles_deg, fbp_angles_deg=None,
                 noise_snr=30, return_fbp=False, image_kind='array', return_volume=False):
        self.directory = directory
        self.name_list = _list_data_files(directory)
        self.image_size = image_size
        self.true_angles_deg = true_angles_deg
        self.fbp_angles_deg = fbp_angles_deg if fbp_angles_deg is not None else true_angles_deg
        self.noise_snr = noise_snr
        self.return_fbp = return_fbp
        self.image_kind = image_kind
        # When True, yield (flattened_image, image_2d) and skip skimage radon —
        # the sinogram is generated downstream (e.g. by the ODL operator).
        self.return_volume = return_volume

    def __len__(self):
        return len(self.name_list)

    def _load_image(self, file_name):
        path = os.path.join(self.directory, file_name)
        if self.image_kind == 'image_file':
            image = imageio.imread(path) / 255.0
            if image.ndim == 3:  # collapse any colour channels to grayscale
                image = image.mean(axis=-1)
            return image
        return np.load(path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Per-sample seed -> reproducible noise (matches original behaviour).
        rng = np.random.RandomState(idx + len(self.name_list))

        image = self._load_image(self.name_list[idx])

        image = torch.tensor(image, dtype=torch.float32)[None, None]
        image = F.interpolate(
            image, size=self.image_size, mode='bilinear',
            antialias=True, align_corners=True)[0, 0].cpu().numpy()

        if self.return_volume:
            # Return the 2D image; the sinogram is generated downstream (ODL).
            image = torch.tensor(image, dtype=torch.float32)
            return image.reshape(-1, 1), image

        sinogram = radon(image, theta=self.true_angles_deg, circle=False)
        sinogram = _add_measurement_noise(sinogram, self.noise_snr, rng)

        image = torch.tensor(image, dtype=torch.float32)

        if self.return_fbp:
            fbp = iradon(sinogram, theta=self.fbp_angles_deg, circle=False)
            fbp = torch.tensor(fbp, dtype=torch.float32)[None, ...]
            return image[None, ...], fbp

        sinogram = torch.tensor(sinogram, dtype=torch.float32)
        return image.reshape(-1, 1), sinogram


def make_dataset(directory, config, true_angles_deg, fbp_angles_deg, network='glimpse'):
    """Build a :class:`RawImageCTDataset` for ``directory`` (raw images).

    The image kind is auto-detected from the files: ``.npy`` arrays vs.
    ``.jpg``/``.png`` images (e.g. the OOD set). ``network`` selects the output:
    ``'glimpse'`` -> ``(flattened_image, sinogram)``, ``'unet'`` -> ``(image,
    fbp)``, ``'odl'`` -> ``(flattened_image, image_2d)`` (sinogram generated by
    the ODL operator downstream).
    """
    names = _list_data_files(directory)
    if not names:
        raise FileNotFoundError(f"No data files found in {directory!r}.")
    ext = os.path.splitext(names[0])[1].lower()

    image_kind = 'image_file' if ext in {'.jpg', '.jpeg', '.png'} else 'array'
    return RawImageCTDataset(
        directory, image_size=config.image_size,
        true_angles_deg=true_angles_deg, fbp_angles_deg=fbp_angles_deg,
        noise_snr=config.noise_snr, return_fbp=(network == 'unet'),
        image_kind=image_kind, return_volume=(network == 'odl'))
