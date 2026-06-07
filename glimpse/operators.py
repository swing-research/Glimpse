"""ODL-based forward/back projectors for GLIMPSE (optional).

This module wraps `ODL <https://odlgroup.github.io/odl/>`_'s 2D parallel-beam
``RayTransform`` (astra GPU backend) so sinograms can be generated with ODL
instead of scikit-image's ``radon``. It is imported lazily — ODL and astra are
only required when ``geometry='odl'``; the default skimage path has no such
dependency.

Geometry (matches ``glimpse.model.GlimpseModel`` with ``geometry='odl'``):
  * image domain ``[-1, 1]^2`` with ``image_size`` pixels per side,
  * projection coordinate ``t = x*cos(theta) + y*sin(theta)``,
  * sinogram shape ``(n_angles, n_det)`` (batched: ``(batch, n_angles, n_det)``),
  * detector depends on ``circle``: ``False`` (default) images the whole square
    with a detector spanning the diagonal ``[-sqrt(2), sqrt(2)]`` over
    ``ceil(sqrt(2)*image_size)`` bins; ``True`` images only the inscribed circle
    (detector ``[-1, 1]``, ``image_size`` bins).
"""

import numpy as np
import torch


class ParallelBeam2DOperator:
    """2D parallel-beam ray transform and FBP backed by ODL + astra.

    Parameters
    ----------
    image_size : int
        Side length of the (square) reconstruction domain.
    angles_rad : array-like
        Projection angles in **radians** (sorted ascending, as ODL requires).
    filter_type : str
        FBP filter for the analytic baseline (e.g. ``'Ram-Lak'``, ``'Hann'``).
    impl : str
        ODL ray-transform backend; ``'astra_cuda'`` (GPU) or ``'astra_cpu'``.
    scale : float
        Optional multiplicative scale applied to generated sinograms (and undone
        in :meth:`fbp`). Defaults to 1.0; the network's learnable filter adapts
        to the absolute scale anyway.
    circle : bool
        ``False`` (default) images the whole square (detector spans the diagonal
        ``[-sqrt(2), sqrt(2)]``, ``ceil(sqrt(2)*image_size)`` bins); ``True``
        images only the inscribed circle (detector ``[-1, 1]``, ``image_size``
        bins). Must match ``GlimpseModel(circle=...)``.
    """

    def __init__(self, image_size, angles_rad, filter_type='Ram-Lak',
                 impl='astra_cuda', scale=1.0, circle=False):
        import odl  # lazy: only needed for the ODL path

        self.image_size = image_size
        self.angles = np.asarray(angles_rad, dtype='float64')
        self.n_angles = len(self.angles)
        self.circle = circle
        det_extent = 1.0 if circle else float(np.sqrt(2))
        self.n_det = image_size if circle else int(np.ceil(image_size * np.sqrt(2)))
        self.scale = scale

        self.reco_space = odl.uniform_discr(
            min_pt=[-1, -1], max_pt=[1, 1], shape=[image_size, image_size], dtype='float32')
        angle_partition = odl.nonuniform_partition(self.angles)
        detector_partition = odl.uniform_partition(-det_extent, det_extent, self.n_det)
        self.geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
        self.op = odl.tomo.RayTransform(self.reco_space, self.geometry, impl=impl)
        self.fbp_op = odl.tomo.fbp_op(self.op, filter_type=filter_type)

    def project(self, volume):
        """Forward project a batch of images to sinograms.

        Parameters
        ----------
        volume : Tensor or ndarray, shape (batch, H, W).

        Returns
        -------
        Tensor, shape ``(batch, n_angles, n_det)`` on the same device as input.
        """
        is_tensor = torch.is_tensor(volume)
        device = volume.device if is_tensor else 'cpu'
        vol_np = (volume.detach().cpu().numpy() if is_tensor else np.asarray(volume)).astype('float32')
        sinos = [np.asarray(self.op(vol_np[i])) for i in range(vol_np.shape[0])]
        sino = np.stack(sinos).astype('float32') * self.scale
        return torch.as_tensor(sino, dtype=torch.float32, device=device)

    def fbp(self, sinogram):
        """Analytic filtered-back-projection baseline.

        Parameters
        ----------
        sinogram : Tensor or ndarray, shape (batch, n_angles, n_det).

        Returns
        -------
        Tensor, shape ``(batch, H, W)`` on the same device as input.
        """
        is_tensor = torch.is_tensor(sinogram)
        device = sinogram.device if is_tensor else 'cpu'
        s_np = (sinogram.detach().cpu().numpy() if is_tensor else np.asarray(sinogram)).astype('float32')
        s_np = s_np / self.scale
        recons = [np.asarray(self.fbp_op(s_np[i])) for i in range(s_np.shape[0])]
        return torch.as_tensor(np.stack(recons).astype('float32'), dtype=torch.float32, device=device)


def build_operator(config, angles_rad):
    """Construct a :class:`ParallelBeam2DOperator` from a Config and angles (rad)."""
    return ParallelBeam2DOperator(
        image_size=config.image_size, angles_rad=angles_rad,
        filter_type='Ram-Lak', impl='astra_cuda', circle=config.circle)
