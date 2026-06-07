"""The GLIMPSE reconstruction network.

GLIMPSE reconstructs a CT image *one pixel at a time*. Given a pixel coordinate
and a sinogram, it predicts that pixel's value from only the sinogram data local
to the pixel — a differentiable analogue of filtered back-projection (FBP):

    1. **Filter** the sinogram in Fourier space (a learnable ramp-style filter).
    2. **Locally sample** the filtered sinogram around the back-projected location
       of the pixel (the "glimpse"), using a learnable sampling patch and a
       learnable sensor geometry (projection angles, detector shift, per-angle
       offsets).
    3. **Predict** the pixel value from the gathered local features with an MLP.

Because prediction is per-coordinate, the model is resolution-agnostic and
generalizes out-of-distribution (e.g. train on faces, test on brain scans).

The learnable parameters are the FBP filter (``fourier_filter``), the sensor
geometry (``theta_rad`` angles, ``z`` per-angle offsets, ``s`` detector shift),
the local sampling pattern (``patch``, ``patch_scale``), and the MLP head
(``MLP``). Short attribute names have descriptive ``@property`` aliases for
readability.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import reflect_coords, build_patch_template, init_fourier_filter

# Optional Hugging Face Hub integration: if `huggingface_hub` is installed,
# GlimpseModel gains `from_pretrained` / `save_pretrained` / `push_to_hub`. If
# not, it falls back to a no-op base so the package imports without the extra
# dependency. (All __init__ args are JSON-serialisable so the mixin can store
# them in config.json — see build_model, which passes the angles as a list.)
try:
    from huggingface_hub import PyTorchModelHubMixin as _HubMixin
except ImportError:  # pragma: no cover - huggingface_hub is optional
    class _HubMixin:  # type: ignore
        pass

# Sinograms are zero-padded to this length before the Fourier-domain filtering
# step (next power of two comfortably above the detector count).
PROJECTION_SIZE_PADDED = 512


class MLPHead(nn.Module):
    """A small fully-connected ReLU network mapping gathered features -> scalar.

    Used as the per-angle-chunk sub-network in the ``multi_MLP`` head variant.
    """

    def __init__(self, in_features, out_features, hidden_features=128):
        super().__init__()
        widths = [hidden_features, hidden_features, hidden_features, out_features]
        layers = []
        prev = in_features
        for width in widths:
            layers.append(nn.Linear(prev, width, bias=True))
            prev = width
        self.fcs = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.fcs[:-1]:
            x = F.relu(layer(x))
        return self.fcs[-1](x)


class GlimpseModel(nn.Module, _HubMixin):
    """Coordinate-based CT reconstruction network (see module docstring).

    Parameters
    ----------
    image_size : int
        Side length of the reconstructed image.
    patch_size : int
        Side of the local sampling patch (``w_size`` in the original code).
    init_angles_deg : np.ndarray
        Projection angles in degrees the model is *told* about (``theta_init``).
        With sensor-geometry learning on, these are refined during training.
    learn_geometry : bool
        Whether the sensor geometry (projection angles + per-angle offsets) is
        learnable (the original ``lsg`` / "learnable sensor geometry" flag).
    learnable_filter : bool
        Whether the Fourier filter is learnable.
    filter_name : str
        Initial filter, e.g. ``'ramp'``, ``'shepp-logan'``, ``'cosine'``.
    head_type : {'MLP', 'multi_MLP'}
        Prediction-head architecture.
    geometry : {'skimage', 'odl'}
        Forward-operator geometry the back-projection must match. ``'odl'``
        (default) matches ODL ``Parallel2dGeometry`` (sinogram ``(batch, n_angles,
        n_det)``; see ``glimpse.operators.ParallelBeam2DOperator``). ``'skimage'``
        matches scikit-image ``radon`` (sinogram ``(batch, n_det, n_angles)``).
    circle : bool
        ODL geometry only. ``False`` (default) images the whole square (detector
        spans the diagonal ``[-sqrt(2), sqrt(2)]``, ``ceil(sqrt(2)*N)`` bins);
        ``True`` images only the inscribed circle (detector ``[-1, 1]``, ``N``
        bins), leaving image corners unmeasured.
    patch_shape : {'round', 'square', 'random'}
        Geometry of the local sampling patch.
    learn_patch : bool
        Whether the sampling patch template is learnable.
    """

    def __init__(self, image_size, patch_size, init_angles_deg, learn_geometry,
                 learnable_filter, filter_name, head_type, patch_shape,
                 learn_patch, geometry='odl', circle=False):
        super().__init__()

        # Accept a list (Hub-serialisable) or ndarray for the angles.
        init_angles_deg = np.asarray(init_angles_deg, dtype=np.float64)

        self.image_size = image_size
        self.patch_size = patch_size
        self.learn_geometry = learn_geometry
        self.learnable_filter = learnable_filter
        self.filter_name = filter_name
        self.n_angles = len(init_angles_deg)
        self.head_type = head_type
        self.geometry = geometry
        self.circle = circle
        self.patch_shape = patch_shape
        self.learn_patch = learn_patch
        if geometry not in ('skimage', 'odl'):
            raise ValueError(f"Unknown geometry {geometry!r}; expected 'skimage' or 'odl'.")

        # Detector layout. skimage's radon and ODL with circle=False both image
        # the whole square: the detector spans the image diagonal (extent
        # sqrt(2)) with ceil(sqrt(2)*N) bins. ODL with circle=True images only
        # the inscribed circle (detector extent 1, N bins). `det_extent` is the
        # detector half-width in image units, used to normalise the ODL
        # back-projection; `recon_extent` is the in-image sampling extent (~N).
        if self.geometry == 'odl' and self.circle:
            self.n_det = image_size
            self.det_extent = 1.0
        else:
            self.n_det = int(np.ceil(image_size * np.sqrt(2)))
            self.det_extent = float(np.sqrt(2))
        self.recon_extent = int(np.floor(self.n_det / np.sqrt(2)))

        # The local patch is patch_size x patch_size sinogram samples.
        self.patch_rows = patch_size
        self.patch_cols = patch_size

        in_features = self.patch_size * self.patch_size * self.n_angles

        if self.head_type == 'multi_MLP':
            n_heads = self.patch_size
            total_features = n_heads * 100
            sub_in = in_features // n_heads
            self.MLP = nn.ModuleList(
                [MLPHead(sub_in, total_features // n_heads, 128) for _ in range(n_heads)]
            )
            self.mixer_MLP = MLPHead(total_features, 1, 128)

        elif self.head_type == 'MLP':
            # Layer widths as powers of two; final width 2**0 = 1 (scalar pixel).
            widths = np.power(2, [8, 8, 8, 8, 7, 7, 7, 6, 6, 0])
            layers = []
            prev = in_features
            for width in widths:
                layers.append(nn.Linear(prev, width, bias=True))
                prev = width
            self.MLP = nn.ModuleList(layers)

        else:
            raise ValueError(f"Unknown head_type {head_type!r}; expected 'MLP' or 'multi_MLP'.")

        # Learnable local sampling-patch template + an overall scale.
        patch = build_patch_template(
            self.patch_shape, self.patch_rows, self.patch_cols, self.image_size)
        self.patch = nn.Parameter(patch.clone().detach(), requires_grad=self.learn_patch)
        self.patch_scale = nn.Parameter(torch.ones(1), requires_grad=True)

        # FBP Fourier filter (optionally learnable).
        fourier_filter = init_fourier_filter(self.filter_name, PROJECTION_SIZE_PADDED)
        self.fourier_filter = nn.Parameter(
            fourier_filter.clone().detach(), requires_grad=self.learnable_filter)

        # Sensor geometry: detector shift `s`, per-angle offsets `z`, angles in
        # radians `theta_rad`. The latter two are learnable iff learn_geometry.
        self.s = nn.Parameter((self.n_det - 1) / 2 * torch.ones(1), requires_grad=True)

        z = (torch.arange(self.n_angles) - (self.n_angles - 1) / 2) / ((self.n_angles - 1) / 2)
        self.z = nn.Parameter(z.clone().detach(), requires_grad=self.learn_geometry)

        theta_rad = torch.deg2rad(
            torch.tensor(init_angles_deg[None, ..., None, None], dtype=torch.float32))
        self.theta_rad = nn.Parameter(
            theta_rad.clone().detach(), requires_grad=self.learn_geometry)

    # ------------------------------------------------------------------ #
    # Readability aliases for the (deliberately short) saved parameters.   #
    # ------------------------------------------------------------------ #
    @property
    def detector_shift(self):
        """Detector half-width `s` mapping image coords to detector position."""
        return self.s

    @property
    def projection_offsets(self):
        """Per-angle normalized offsets `z` into the sinogram angle axis."""
        return self.z

    @property
    def angles_rad(self):
        """Projection angles in radians `theta_rad` (learnable geometry)."""
        return self.theta_rad

    # ------------------------------------------------------------------ #
    def extract_sin(self, coords, sinogram):
        """Back-project: sample the (filtered) sinogram at image coordinates.

        Dispatches to the back-projection geometry matching ``self.geometry``.
        For each image coordinate it computes, per angle, the detector location
        the ray through that coordinate hits, and reads the sinogram there via
        bilinear ``grid_sample``. Returns ``(batch, n_angles, n_coords)``.
        """
        if self.geometry == 'odl':
            return self._extract_sin_odl(coords, sinogram)
        return self._extract_sin_skimage(coords, sinogram)

    def _extract_sin_skimage(self, coords, sinogram):
        """Back-projection for skimage ``radon`` geometry.

        skimage's sinogram is laid out ``(batch, n_det, n_angles)`` with the
        detector spanning the image diagonal; the projection coordinate is
        ``t = y*cos(theta) - x*sin(theta)`` in centered pixel units.
        """
        batch = coords.shape[0]
        recon_extent = self.recon_extent

        # Reflect coordinates into the valid image extent, then renormalize.
        coords = reflect_coords((coords + 0.5) * (recon_extent - 1), -0.5, recon_extent - 1 + 0.5)
        coords = coords / (recon_extent - 1) - 0.5

        # Treat the sinogram as an image of shape (angle, detector) to sample.
        sinogram_grid = sinogram.permute(0, 2, 1).unsqueeze(1)
        coords = coords.unsqueeze(1) * (recon_extent - 1)
        x_coord = coords[:, :, :, 0]
        y_coord = coords[:, :, :, 1]

        y_coord = y_coord / self.s
        x_coord = x_coord / self.s
        x_coord = x_coord.unsqueeze(1).repeat(1, self.n_angles, 1, 1)
        y_coord = y_coord.unsqueeze(1).repeat(1, self.n_angles, 1, 1)

        # Radon projection coordinate per angle (the "t" axis of the sinogram).
        proj_t = y_coord * torch.cos(self.theta_rad) - x_coord * torch.sin(self.theta_rad)
        proj_t = proj_t[..., None]

        # Pair each projection coordinate with its per-angle offset z.
        z = self.z[..., None, None, None]
        z = z[None, ...].repeat(proj_t.shape[0], 1, proj_t.shape[2], proj_t.shape[3], 1)
        grid = torch.concat((proj_t, z), dim=-1)
        grid = grid.reshape(batch, self.n_angles * grid.shape[2], grid.shape[3], 2)

        back_projection = F.grid_sample(sinogram_grid, grid, align_corners=True, mode='bilinear')
        back_projection = back_projection.reshape(batch, self.n_angles, grid.shape[2])
        return back_projection

    def _extract_sin_odl(self, coords, sinogram):
        """Back-projection for ODL ``Parallel2dGeometry``.

        ODL's sinogram is laid out ``(batch, n_angles, n_det)`` with the image
        on ``[-1, 1]^2`` and a detector of ``n_det = image_size`` bins on
        ``[-1, 1]``. The projection coordinate is ``t = x*cos(theta) +
        y*sin(theta)`` (x = axis 0, y = axis 1), verified empirically against
        ``odl.tomo.RayTransform``. No coordinate reflection is applied: rays from
        image corners that miss the detector correctly sample zero.
        """
        batch = coords.shape[0]
        n_det = sinogram.shape[2]

        # Sinogram is already (batch, n_angles, n_det) -> sample as an image.
        sinogram_grid = sinogram.unsqueeze(1)
        coords = coords.unsqueeze(1)
        # Model coords are [-0.5, 0.5]; ODL image domain is [-1, 1].
        x_coord = (coords[:, :, :, 0] * 2.0).unsqueeze(1).repeat(1, self.n_angles, 1, 1)
        y_coord = (coords[:, :, :, 1] * 2.0).unsqueeze(1).repeat(1, self.n_angles, 1, 1)

        # Detector coordinate per angle, mapped to align_corners grid units.
        # Dividing by det_extent accounts for a detector spanning [-extent, extent]
        # (sqrt(2) for the whole image / circle=False, 1 for circle=True).
        proj_t = x_coord * torch.cos(self.theta_rad) + y_coord * torch.sin(self.theta_rad)
        det_coord = (proj_t / self.det_extent * n_det / 2.0) / self.s
        det_coord = det_coord[..., None]

        # Pair each detector coordinate with its per-angle offset z.
        z = self.z[..., None, None, None]
        z = z[None, ...].repeat(det_coord.shape[0], 1, det_coord.shape[2], det_coord.shape[3], 1)
        grid = torch.concat((det_coord, z), dim=-1)
        grid = grid.reshape(batch, self.n_angles * grid.shape[2], grid.shape[3], 2)

        back_projection = F.grid_sample(sinogram_grid, grid, align_corners=True, mode='bilinear')
        back_projection = back_projection.reshape(batch, self.n_angles, grid.shape[2])
        return back_projection

    def gather_local_glimpse(self, sinogram, coordinate):
        """Gather the local sinogram patch (the "glimpse") for each pixel.

        Adds the learnable patch template around every pixel coordinate and
        back-projects the whole set, returning gathered features of shape
        ``(batch * n_pixels, n_angles, patch_rows, patch_cols)``.
        """
        batch = sinogram.shape[0]
        recon_extent = self.recon_extent
        n_pixels = coordinate.shape[1]
        coordinate = coordinate * 2

        patch = self.patch_scale * self.patch / (recon_extent / self.image_size)
        patch = patch[None, None]
        n_rows = self.patch_rows
        n_cols = self.patch_cols

        coordinate = coordinate.unsqueeze(2).unsqueeze(2)
        sample_coords = coordinate + patch
        sample_coords = sample_coords.reshape(batch, n_pixels * n_rows, n_cols, 2)
        sample_coords = sample_coords.reshape(batch, n_pixels * n_rows * n_cols, 2)

        samples = self.extract_sin(sample_coords / 2, sinogram)
        samples = samples.reshape(batch, -1, n_pixels * n_rows, n_cols)
        samples = samples.permute(0, 2, 3, 1)
        samples = samples.reshape(batch, n_pixels, n_rows, n_cols, self.n_angles)
        samples = samples.reshape(batch * n_pixels, n_rows, n_cols, self.n_angles)
        samples = samples.permute(0, 3, 1, 2)
        return samples

    def forward(self, coordinate, sinogram):
        """Predict pixel value(s) at ``coordinate`` from ``sinogram``.

        Parameters
        ----------
        coordinate : Tensor, shape (batch, n_pixels, 2)
            Pixel coordinates normalized to ``[-0.5, 0.5]``.
        sinogram : Tensor
            ``(batch, n_det, n_angles)`` for ``geometry='skimage'`` or
            ``(batch, n_angles, n_det)`` for ``geometry='odl'``.

        Returns
        -------
        Tensor, shape (batch, n_pixels, 1) of predicted pixel values.
        """
        # Step 1: Fourier-domain filtering (the FBP filter, learnable). The
        # detector axis differs by geometry (skimage: dim 1, ODL: dim 2).
        if self.geometry == 'odl':
            n_det = sinogram.shape[2]
            padded_sinogram = F.pad(sinogram, (0, PROJECTION_SIZE_PADDED - n_det))
            spectrum = torch.fft.fft(padded_sinogram, dim=2) * self.fourier_filter.reshape(1, 1, -1)
            filtered_sinogram = torch.fft.ifft(spectrum, dim=2)[:, :, :n_det].real
        else:
            n_det = sinogram.shape[1]
            padded_sinogram = F.pad(sinogram, (0, 0, 0, PROJECTION_SIZE_PADDED - n_det))
            spectrum = torch.fft.fft(padded_sinogram, dim=1) * self.fourier_filter
            filtered_sinogram = torch.fft.ifft(spectrum, dim=1)[:, :n_det].real

        batch, n_pixels, _ = coordinate.shape

        # Step 2: gather the local glimpse around each pixel.
        local_features = self.gather_local_glimpse(filtered_sinogram, coordinate)

        # Step 3: predict the pixel value with the chosen head.
        if self.head_type == 'multi_MLP':
            chunk_outs = []
            for i, head in enumerate(self.MLP):
                chunk = torch.flatten(local_features[:, :, :, i], 1)
                chunk_outs.append(head(chunk))
            x = torch.cat(chunk_outs, dim=1)
            x = self.mixer_MLP(x)
        else:  # 'MLP'
            x = torch.flatten(local_features, 1)
            for layer in self.MLP[:-1]:
                x = F.relu(layer(x))
            x = self.MLP[-1](x)

        x = x.reshape(batch, n_pixels, -1)
        x = x * np.pi / 2
        return x
