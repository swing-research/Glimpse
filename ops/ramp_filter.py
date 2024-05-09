"""
This script is used to break the filtering operator in odl so that the ramp filter can be made trainable

"""

from odl.discr import ResizingOperator
from odl.trafos import FourierTransform, PYFFTW_AVAILABLE
from odl.tomo.analytic.filtered_back_projection import _fbp_filter,_axis_in_detector,_rotation_direction_in_detector
import numpy as np



def fbp_filter_op_custom(ray_trafo, padding=True, filter_type='Ram-Lak',
                  frequency_scaling=1.0):
    """Create a filter operator for FBP from a `RayTransform`.

    Parameters
    ----------
    ray_trafo : `RayTransform`
        The ray transform (forward operator) whose approximate inverse should
        be computed. Its geometry has to be any of the following

        `Parallel2dGeometry` : Exact reconstruction

        `Parallel3dAxisGeometry` : Exact reconstruction

        `FanBeamGeometry` : Approximate reconstruction, correct in limit of
        fan angle = 0.
        Only flat detectors are supported (det_curvature_radius is None).

        `ConeBeamGeometry`, pitch = 0 (circular) : Approximate reconstruction,
        correct in the limit of fan angle = 0 and cone angle = 0.

        `ConeBeamGeometry`, pitch > 0 (helical) : Very approximate unless a
        `tam_danielson_window` is used. Accurate with the window.

        Other geometries: Not supported

    padding : bool, optional
        If the data space should be zero padded. Without padding, the data may
        be corrupted due to the circular convolution used. Using padding makes
        the algorithm slower.
    filter_type : optional
        The type of filter to be used.
        The predefined options are, in approximate order from most noise
        senstive to least noise sensitive:
        ``'Ram-Lak'``, ``'Shepp-Logan'``, ``'Cosine'``, ``'Hamming'`` and
        ``'Hann'``.
        A callable can also be provided. It must take an array of values in
        [0, 1] and return the filter for these frequencies.
    frequency_scaling : float, optional
        Relative cutoff frequency for the filter.
        The normalized frequencies are rescaled so that they fit into the range
        [0, frequency_scaling]. Any frequency above ``frequency_scaling`` is
        set to zero.

    Returns
    -------
    filter_op : `Operator`
        Filtering operator for FBP based on ``ray_trafo``.

    See Also
    --------
    tam_danielson_window : Windowing for helical data
    """
    impl = 'pyfftw' if PYFFTW_AVAILABLE else 'numpy'
    alen = ray_trafo.geometry.motion_params.length

    if ray_trafo.domain.ndim == 2:
        # Define ramp filter
        def fourier_filter(x):
            abs_freq = np.abs(x[1])
            norm_freq = abs_freq / np.max(abs_freq)
            filt = _fbp_filter(norm_freq, filter_type, frequency_scaling)
            scaling = 1 / (2 * alen)
            return filt * np.max(abs_freq) * scaling

        # Define (padded) fourier transform
        if padding:
            # Define padding operator
            ran_shp = (ray_trafo.range.shape[0],
                       ray_trafo.range.shape[1] * 2 - 1)
            resizing = ResizingOperator(ray_trafo.range, ran_shp=ran_shp)

            fourier = FourierTransform(resizing.range, axes=1, impl=impl)
            fourier = fourier * resizing
        else:
            fourier = FourierTransform(ray_trafo.range, axes=1, impl=impl)

    elif ray_trafo.domain.ndim == 3:
        # Find the direction that the filter should be taken in
        rot_dir = _rotation_direction_in_detector(ray_trafo.geometry)

        # Find what axes should be used in the fourier transform
        used_axes = (rot_dir != 0)
        if used_axes[0] and not used_axes[1]:
            axes = [1]
        elif not used_axes[0] and used_axes[1]:
            axes = [2]
        else:
            axes = [1, 2]

        # Add scaling for cone-beam case
        if hasattr(ray_trafo.geometry, 'src_radius'):
            scale = (ray_trafo.geometry.src_radius
                     / (ray_trafo.geometry.src_radius
                        + ray_trafo.geometry.det_radius))

            if ray_trafo.geometry.pitch != 0:
                # In helical geometry the whole volume is not in each
                # projection and we need to use another weighting.
                # Ideally each point in the volume effects only
                # the projections in a half rotation, so we assume that that
                # is the case.
                scale *= alen / (np.pi)
        else:
            scale = 1.0

        # Define ramp filter
        def fourier_filter(x):
            # If axis is aligned to a coordinate axis, save some memory and
            # time by using broadcasting
            if not used_axes[0]:
                abs_freq = np.abs(rot_dir[1] * x[2])
            elif not used_axes[1]:
                abs_freq = np.abs(rot_dir[0] * x[1])
            else:
                abs_freq = np.abs(rot_dir[0] * x[1] + rot_dir[1] * x[2])
            norm_freq = abs_freq / np.max(abs_freq)
            filt = _fbp_filter(norm_freq, filter_type, frequency_scaling)
            scaling = scale * np.max(abs_freq) / (2 * alen)
            print(filt.shape)
            return filt * scaling

        # Define (padded) fourier transform
        if padding:
            # Define padding operator
            if used_axes[0]:
                padded_shape_u = ray_trafo.range.shape[1] * 2 - 1
            else:
                padded_shape_u = ray_trafo.range.shape[1]

            if used_axes[1]:
                padded_shape_v = ray_trafo.range.shape[2] * 2 - 1
            else:
                padded_shape_v = ray_trafo.range.shape[2]

            ran_shp = (ray_trafo.range.shape[0],
                       padded_shape_u,
                       padded_shape_v)
            resizing = ResizingOperator(ray_trafo.range, ran_shp=ran_shp)

            fourier = FourierTransform(resizing.range, axes=axes, impl=impl)
            fourier = fourier * resizing
        else:
            fourier = FourierTransform(ray_trafo.range, axes=axes, impl=impl)
    else:
        raise NotImplementedError('FBP only implemented in 2d and 3d')

    # Create ramp in the detector direction
    ramp_function = fourier.range.element(fourier_filter)

    weight = 1
    if not ray_trafo.range.is_weighted:
        # Compensate for potentially unweighted range of the ray transform
        weight *= ray_trafo.range.cell_volume

    if not ray_trafo.domain.is_weighted:
        # Compensate for potentially unweighted domain of the ray transform
        weight /= ray_trafo.domain.cell_volume

    ramp_function *= weight

    # Create ramp filter via the convolution formula with fourier transforms
    return  ramp_function ,fourier



if __name__ == "__main__":
    """
    Test the outputs
    """
    import torch
    import numpy as np