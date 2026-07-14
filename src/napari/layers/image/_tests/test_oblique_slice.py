import warnings

import numpy as np
import pytest

from napari.components.dims import Dims
from napari.layers import Image, Labels

N = 30
THETA = 30.0
POINT0 = 15.0
# Wide enough to not clamp POINT0 -- Dims defaults to range (0, 2, 1) per
# dim, which silently clips any point outside it.
WIDE_RANGE = ((0, 60, 1),) * 3


def rotation_in_axes(ndim, axes, angle_degrees):
    """A rotation matrix rotating only within the given pair of axes."""
    i, j = axes
    matrix = np.eye(ndim)
    c, s = np.cos(np.radians(angle_degrees)), np.sin(np.radians(angle_degrees))
    matrix[i, i] = c
    matrix[i, j] = -s
    matrix[j, i] = s
    matrix[j, j] = c
    return matrix


# Rotates data axes (0, 1) -- mixes the not-displayed axis (0) with a
# displayed axis (1) when slicing with order=(0, 1, 2), ndisplay=2, which
# is exactly the oblique case (solid, non-orthogonal).
ROTATE_01 = rotation_in_axes(3, (0, 1), THETA)


def ramp_data(axis: int) -> np.ndarray:
    """A 3D array whose value at each voxel is its coordinate along `axis`."""
    grids = np.meshgrid(*(np.arange(N),) * 3, indexing='ij')
    return grids[axis].astype(float)


def slice_ramp_layer(layer, point0=POINT0):
    layer._slice_dims(
        Dims(ndim=3, ndisplay=2, point=(point0, 0, 0), range=WIDE_RANGE)
    )
    return np.asarray(layer._slice.image.view)


def expected_value_and_mask(layer, out_shape, axis, point0=POINT0):
    """Ground truth for `ramp_data(axis)` sliced obliquely by `layer`.

    Reimplements nothing from the slicing pipeline except reusing
    `slice_plane`, which is independently covered by
    `test_slice_input.py`, to know which canvas pixels land in bounds.
    The actual expected *value* is a closed-form trig formula.
    """
    slice_input = layer._slice_input
    world_to_data = layer._data_to_world.inverse
    plane = slice_input.slice_plane(world_to_data)

    # Must match _oblique_canvas_grid, which samples at pixel *centers*
    # (the unaugmented extent) -- the augmented, pixel-edge extent is for
    # drawing a box around the grid, not for the grid itself.
    world_extent = layer.extent.world
    origin = world_extent[0, list(slice_input.displayed)]
    step = float(np.min(np.abs(layer._data_to_world.scale)))

    expected = np.zeros(out_shape)
    in_bounds = np.zeros(out_shape, dtype=bool)
    theta = np.radians(THETA)
    for i in range(out_shape[0]):
        for j in range(out_shape[1]):
            u = origin[0] + i * step
            v = origin[1] + j * step
            data_coord = plane(np.array([u, v]))
            in_bounds[i, j] = bool(
                np.all(data_coord >= 0) and np.all(data_coord <= N - 1)
            )
            if axis == 2:
                # Untouched by ROTATE_01, so data axis 2 == world v exactly.
                expected[i, j] = v
            elif axis == 1:
                expected[i, j] = -np.sin(theta) * point0 + np.cos(theta) * u
    return expected, in_bounds


@pytest.mark.parametrize('axis', [1, 2])
def test_oblique_slice_matches_closed_form(axis):
    data = ramp_data(axis)
    layer = Image(data, rotate=ROTATE_01)
    out = slice_ramp_layer(layer)

    expected, in_bounds = expected_value_and_mask(layer, out.shape, axis)

    assert in_bounds.sum() > 0
    np.testing.assert_allclose(out[in_bounds], expected[in_bounds], atol=1e-8)


def test_oblique_slice_masks_out_of_bounds_as_zero():
    data = ramp_data(2)
    layer = Image(data, rotate=ROTATE_01)
    out = slice_ramp_layer(layer)

    _, in_bounds = expected_value_and_mask(layer, out.shape, 2)
    assert (~in_bounds).sum() > 0
    np.testing.assert_allclose(out[~in_bounds], 0)


def test_oblique_slice_no_warning():
    data = ramp_data(2)
    layer = Image(data, rotate=ROTATE_01)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        slice_ramp_layer(layer)


def test_labels_oblique_slice_uses_nearest_neighbor():
    # A checkerboard-ish label pattern: linear interpolation (order=1)
    # would invent label values between 0 and 1 that never existed.
    data = (ramp_data(2).astype(int) % 2).astype(np.int32)
    layer = Labels(data, rotate=ROTATE_01)
    slice_ramp_layer(layer)
    displayed_values = np.unique(np.asarray(layer._slice.image.raw))
    assert set(displayed_values).issubset({0, 1})


def test_sheared_transform_still_falls_back_and_warns():
    # Out of v1 scope (shear): should still hit the old axis-aligned
    # fallback and warn, exactly like before this feature existed.
    # shear[1, 0] (not [0, 1]) so that world_to_data mixes the
    # not-displayed world axis (0) into a displayed data axis (1) --
    # the specific direction is_orthogonal checks.
    data = ramp_data(2)
    shear = np.eye(3)
    shear[1, 0] = 0.5
    with pytest.warns(UserWarning, match='Non-orthogonal slicing'):
        Image(data, shear=shear)


def test_multiscale_oblique_still_falls_back_and_warns():
    # Out of v1 scope (multiscale): should still hit the old fallback.
    data = ramp_data(2)
    with pytest.warns(UserWarning, match='Non-orthogonal slicing'):
        Image([data, data[::2, ::2, ::2]], rotate=ROTATE_01)
