import numpy as np
import pytest

from napari.layers.utils._slice_input import _SliceInput, _ThickNDSlice
from napari.utils.transforms import Affine


def make_slice_input(ndim, ndisplay, point=None) -> _SliceInput:
    return _SliceInput(
        ndisplay=ndisplay,
        world_slice=_ThickNDSlice.make_full(point=point, ndim=ndim),
        order=tuple(range(ndim)),
    )


def rotation_in_axes(ndim, axes, angle_degrees):
    """A rotation matrix rotating only within the given pair of axes.

    Avoids relying on the yaw/pitch/roll convention `Affine(rotate=...)`
    uses for 3-tuples, which does not map angle i to a rotation about
    axis i.
    """
    i, j = axes
    matrix = np.eye(ndim)
    c, s = np.cos(np.radians(angle_degrees)), np.sin(np.radians(angle_degrees))
    matrix[i, i] = c
    matrix[i, j] = -s
    matrix[j, i] = s
    matrix[j, j] = c
    return matrix


def test_is_solid_true_for_identity():
    slice_input = make_slice_input(ndim=3, ndisplay=2)
    assert slice_input.is_solid(Affine(ndim=3))


def test_is_solid_true_for_rotation_and_scale():
    world_to_data = Affine(
        ndim=3, rotate=(0, 0, 30), scale=(1, 2, 3), translate=(1, -2, 3)
    ).inverse
    slice_input = make_slice_input(ndim=3, ndisplay=2)
    assert slice_input.is_solid(world_to_data)


def test_is_solid_false_for_shear():
    world_to_data = Affine(ndim=3, shear=(0.5, 0, 0))
    slice_input = make_slice_input(ndim=3, ndisplay=2)
    assert not slice_input.is_solid(world_to_data)


def test_is_solid_true_implies_is_orthogonal_for_axis_aligned_rotation():
    # order=(0, 1, 2), ndisplay=2 -> displayed=(1, 2), not_displayed=(0,).
    # Rotating purely within the displayed subspace (1, 2) keeps the
    # not-displayed axis fixed, so the slice is both solid and orthogonal.
    slice_input = make_slice_input(ndim=3, ndisplay=2)
    world_to_data = Affine(
        ndim=3, linear_matrix=rotation_in_axes(3, (1, 2), 30)
    )
    assert slice_input.is_solid(world_to_data)
    assert slice_input.is_orthogonal(world_to_data)


def test_is_solid_and_non_orthogonal_for_in_plane_rotation():
    # A rotation mixing the not-displayed axis (0) with a displayed axis
    # (1) is solid but not orthogonal -- exactly the case oblique slicing
    # targets.
    slice_input = make_slice_input(ndim=3, ndisplay=2)
    world_to_data = Affine(
        ndim=3, linear_matrix=rotation_in_axes(3, (0, 1), 30)
    )
    assert slice_input.is_solid(world_to_data)
    assert not slice_input.is_orthogonal(world_to_data)


@pytest.mark.parametrize(
    'rotate', [(0, 0, 0), (12, -34, 56), (90, 0, 0), (0, 45, 0)]
)
def test_slice_plane_matches_direct_transform(rotate):
    ndim = 3
    point = (5.0, 0.0, 0.0)
    world_to_data = Affine(
        ndim=ndim, rotate=rotate, scale=(1, 2, 0.5), translate=(3, -1, 2)
    ).inverse
    slice_input = make_slice_input(ndim=ndim, ndisplay=2, point=point)
    assert slice_input.is_solid(world_to_data)

    plane = slice_input.slice_plane(world_to_data)

    rng = np.random.default_rng(0)
    canvas_coords = rng.uniform(-10, 10, size=(20, 2))

    world_coords = np.zeros((20, ndim))
    world_coords[:, slice_input.displayed] = canvas_coords
    world_coords[:, slice_input.not_displayed] = [
        point[d] for d in slice_input.not_displayed
    ]
    expected = world_to_data(world_coords)

    np.testing.assert_allclose(plane(canvas_coords), expected, atol=1e-10)


def test_slice_plane_offset_matches_data_slice_point_when_orthogonal():
    # When the transform is orthogonal, slice_plane's offset in the
    # non-displayed dims should agree with data_slice's point.
    ndim = 3
    point = (5.0, 0.0, 0.0)
    # order=(0, 1, 2), ndisplay=2 -> displayed=(1, 2); rotating only
    # within (1, 2) keeps the not-displayed axis (0) orthogonal.
    linear_matrix = rotation_in_axes(ndim, (1, 2), 30) @ np.diag(
        (1.0, 2.0, 3.0)
    )
    world_to_data = Affine(
        ndim=ndim, linear_matrix=linear_matrix, translate=(1, -2, 3)
    ).inverse
    slice_input = make_slice_input(ndim=ndim, ndisplay=2, point=point)
    assert slice_input.is_orthogonal(world_to_data)

    plane = slice_input.slice_plane(world_to_data)
    thick_slice = slice_input.data_slice(world_to_data)

    for d in slice_input.not_displayed:
        assert plane.offset[d] == pytest.approx(thick_slice.point[d])


def test_slice_plane_shape_and_call():
    ndim = 4
    slice_input = make_slice_input(ndim=ndim, ndisplay=2)
    world_to_data = Affine(ndim=ndim)
    plane = slice_input.slice_plane(world_to_data)

    assert plane.matrix.shape == (ndim, 2)
    assert plane.offset.shape == (ndim,)
    assert plane.ndim == ndim
    assert plane.displayed == tuple(slice_input.displayed)
    assert plane.not_displayed == tuple(slice_input.not_displayed)

    canvas_coords = np.array([[1.0, 2.0], [3.0, 4.0]])
    data_coords = plane(canvas_coords)
    assert data_coords.shape == (2, ndim)
