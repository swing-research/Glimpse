"""GLIMPSE: Generalized Locality for Scalable and Robust CT.

Coordinate-based CT reconstruction from sparse-view sinograms (arXiv:2401.00816).

Common entry points::

    from glimpse import Config, GlimpseModel, build_model, load_checkpoint
    from glimpse import RawImageCTDataset, reconstruct_image, make_coordinate_grid
"""

from .config import Config
from .data import RawImageCTDataset, make_dataset
from .engine import build_model, evaluate, get_device, load_checkpoint, train
from .metrics import PSNR, SSIM, count_parameters, psnr, ssim
from .model import GlimpseModel, MLPHead
from .operators import ParallelBeam2DOperator, build_operator
from .reconstruct import fbp_batch, make_coordinate_grid, reconstruct_image

__version__ = "0.1.0"

__all__ = [
    "Config",
    "GlimpseModel",
    "MLPHead",
    "RawImageCTDataset",
    "make_dataset",
    "build_model",
    "load_checkpoint",
    "train",
    "evaluate",
    "get_device",
    "make_coordinate_grid",
    "reconstruct_image",
    "fbp_batch",
    "ParallelBeam2DOperator",
    "build_operator",
    "psnr",
    "ssim",
    "PSNR",
    "SSIM",
    "count_parameters",
    "__version__",
]
