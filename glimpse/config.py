"""Experiment configuration for GLIMPSE.

A single :class:`Config` dataclass holds every hyperparameter and path. It can be
loaded from a YAML file (see ``configs/``) and overridden from the command line,
replacing the old global-singleton ``config.py``::

    python train.py --config configs/lodopab.yaml --n-angles 60 --epochs 5000

All fields are plain values, so a ``Config`` is trivially serialisable and a
config file is a faithful, self-documenting record of an experiment.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a listed dependency
    yaml = None


@dataclass
class Config:
    # --- Acquisition geometry (must match the dataset) ---
    image_size: int = 128          # reconstructed image side length, in pixels
    n_angles: int = 30             # number of projection angles (views) acquired
    noise_snr: float = 30          # measurement noise level, signal-to-noise ratio in dB
    # Forward operator used to simulate sinograms. Options:
    #   'odl'     : ODL Parallel2dGeometry + astra_cuda (GPU). Recommended/default;
    #               matches the published glimpse.pt. Needs environment-odl.yml.
    #   'skimage' : scikit-image radon (CPU). No GPU/astra needed; train your own.
    geometry: str = 'odl'
    # Reconstruction support (ODL geometry only). False (default): image the whole
    # square — the detector spans the image diagonal [-sqrt(2), sqrt(2)] with
    # ceil(sqrt(2)*image_size) bins. True: only the inscribed circle (detector
    # [-1, 1], image_size bins), so image corners are not measured.
    circle: bool = False

    # --- Sensor (mis)calibration ---
    # How the projection angles the model is *told* (theta_init) diverge from the
    # *true* acquisition angles (theta_actual). Options:
    #   'No'     : calibrated — told angles == true angles.
    #   'random' : true angles jittered by Gaussian noise.
    #   'fixed'  : a constant angular offset is added.
    #   'blind'  : angles unknown — told random angles (learn_geometry recovers them).
    uncalibrated_type: str = 'No'

    # --- Model ---
    patch_size: int = 9            # side length of the local sampling 'glimpse' patch
    head_type: str = 'MLP'         # prediction head: 'MLP' or 'multi_MLP' (per-angle)
    filter_name: str = 'ramp'      # FBP filter init: 'ramp'|'shepp-logan'|'cosine'|'hamming'|'hann'
    learnable_filter: bool = True  # let training adapt the Fourier (FBP) filter
    learn_geometry: bool = True    # learn the sensor geometry (angles + per-angle offsets)
    patch_shape: str = 'round'     # local sampling pattern: 'round'|'square'|'random'
    learn_patch: bool = True       # learn the local sampling-patch point pattern

    # --- Training ---
    epochs: int = 3000             # number of training epochs
    batch_size: int = 64           # images per mini-batch
    lr: float = 1e-4               # Adam learning rate
    # GLIMPSE predicts one pixel at a time, so each step optimises a random
    # subset of pixels rather than whole images:
    pixels_per_step: int = 512     # number of random pixels optimised per gradient step
    steps_per_batch: int = 3       # gradient steps taken on each image mini-batch
    train: bool = True             # if False, only evaluate (no training)
    restore_model: bool = True     # reload the checkpoint (if present) before training
    num_workers: int = 8           # DataLoader worker processes (match allocated CPUs)

    # --- Data paths (raw image folders; sinograms are rendered on the fly) ---
    train_path: str = 'datasets/train'
    test_path: str = 'datasets/test'
    ood_path: str = 'datasets/ood'   # out-of-distribution set for generalization checks

    # --- Experiment / evaluation ---
    exp_desc: str = 'glimpse'      # suffix for the experiment dir name
    gpu: int = 0                   # CUDA device index
    ood_analysis: bool = True      # also evaluate on the OOD set
    sample_number: int = 1         # number of samples drawn in the saved visualisations
    cmap: str = 'gray'             # matplotlib colormap for the output PNGs

    # ------------------------------------------------------------------ #
    @property
    def exp_path(self) -> str:
        """Per-experiment directory: ``experiments/<size>_<angles>_<desc>``."""
        return f'experiments/{self.image_size}_{self.n_angles}_{self.exp_desc}'

    def resolve_angles(self):
        """Return ``(true_angles_deg, init_angles_deg)`` for this config.

        ``true_angles_deg`` are the real projection angles; ``init_angles_deg``
        are what the model is initialised/told, diverging per
        ``uncalibrated_type``. Mirrors the original config.py logic, including
        its fixed RNG seed so results are reproducible.
        """
        true_angles = np.linspace(0.0, 180.0, self.n_angles, endpoint=False)
        rng = np.random.RandomState(2)

        if self.uncalibrated_type == 'No':
            init_angles = true_angles.copy()
        elif self.uncalibrated_type == 'random':
            shifts = rng.randn(self.n_angles) * 2.0
            init_angles = np.linspace(0.0, 180.0, self.n_angles, endpoint=False) + shifts
        elif self.uncalibrated_type == 'fixed':
            init_angles = np.linspace(3.0, 183.0, self.n_angles, endpoint=False)
        elif self.uncalibrated_type == 'blind':
            init_angles = np.sort(rng.rand(self.n_angles) * 180.0)
        else:
            raise ValueError(
                f"Unknown uncalibrated_type {self.uncalibrated_type!r}; "
                "expected 'No', 'random', 'fixed' or 'blind'.")

        return true_angles, init_angles

    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        if yaml is None:
            raise ImportError("pyyaml is required to load YAML configs (pip install pyyaml).")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_args(cls, argv=None) -> "Config":
        """Build a Config from ``--config <yaml>`` plus optional CLI overrides.

        Any dataclass field can be overridden with ``--field-name value`` (dashes
        in place of underscores), e.g. ``--n-angles 60 --learn-geometry false``.
        """
        parser = argparse.ArgumentParser(description="GLIMPSE training / evaluation.")
        parser.add_argument('--config', type=str, default=None,
                            help='Path to a YAML config file (defaults applied otherwise).')
        for f in fields(cls):
            flag = '--' + f.name.replace('_', '-')
            if f.type == 'bool' or f.type is bool:
                parser.add_argument(flag, type=_str2bool, default=None)
            elif f.type == 'int' or f.type is int:
                parser.add_argument(flag, type=int, default=None)
            elif f.type == 'float' or f.type is float:
                parser.add_argument(flag, type=float, default=None)
            else:
                parser.add_argument(flag, type=str, default=None)
        args = parser.parse_args(argv)

        config = cls.from_yaml(args.config) if args.config else cls()
        for f in fields(cls):
            val = getattr(args, f.name)
            if val is not None:
                setattr(config, f.name, val)
        return config


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {'1', 'true', 'yes', 'y', 't'}
