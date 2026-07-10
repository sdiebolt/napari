from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from napari.utils.transforms import Affine

if TYPE_CHECKING:
    import numpy.typing as npt

    from napari.layers.utils._slice_input import _PlaneSlice
    from napari.types import ArrayLike


class ObliqueCanvasGrid(NamedTuple):
    """A regular, world-axis-aligned sampling grid for an oblique slice.

    Attributes
    ----------
    shape : 2-tuple of int
        The number of samples along each displayed dim.
    world_origin : (2,) array
        The world coordinate of sample (0, 0).
    world_step : float
        The world-space spacing between adjacent samples, uniform across
        both displayed dims.
    """

    shape: tuple[int, int]
    world_origin: npt.NDArray
    world_step: float


def canvas_corners_world(grid: ObliqueCanvasGrid) -> npt.NDArray:
    """World coordinates of the 4 corners of a canvas grid, shape (4, 2)."""
    max_index = np.array(grid.shape) - 1
    corner_indices = np.array(
        [[0, 0], [0, max_index[1]], [max_index[0], 0], max_index]
    )
    return grid.world_origin + corner_indices * grid.world_step


def canvas_world_grid(grid: ObliqueCanvasGrid) -> npt.NDArray:
    """World coordinates of every sample in a canvas grid, shape (*shape, 2)."""
    indices = np.stack(
        np.meshgrid(*(np.arange(n) for n in grid.shape), indexing='ij'),
        axis=-1,
    )
    return grid.world_origin + indices * grid.world_step


def data_bounding_box(
    plane: _PlaneSlice,
    corners_world: npt.NDArray,
    data_shape: tuple[int, ...],
    margin: int,
) -> npt.NDArray:
    """Axis-aligned data-space bounding box, shape (2, ndim), covering `corners_world`.

    Clips to `data_shape` and pads by `margin` samples on each side to
    leave room for interpolation at the edges.
    """
    data_corners = plane(corners_world)
    max_index = np.array(data_shape) - 1
    lo = np.clip(
        np.floor(data_corners.min(axis=0)).astype(int) - margin, 0, max_index
    )
    hi = np.clip(
        np.ceil(data_corners.max(axis=0)).astype(int) + margin, 0, max_index
    )
    return np.stack([lo, hi])


def resample_oblique_plane(
    data: ArrayLike,
    plane: _PlaneSlice,
    grid: ObliqueCanvasGrid,
    *,
    order: int,
    cval: float,
) -> npt.NDArray:
    """Resamples the oblique 2D plane described by `plane` and `grid` through nD `data`.

    Extracts an axis-aligned bounding box around the plane's footprint
    using plain slicing (works for numpy/dask/zarr arrays alike), then
    resamples that materialized chunk onto `grid` with
    `scipy.ndimage.map_coordinates`.
    """
    from scipy import ndimage as ndi

    # order + 1 samples are needed on each side of a query point for
    # spline interpolation of that order; +1 more as a safety margin.
    margin = order + 2
    bbox = data_bounding_box(
        plane, canvas_corners_world(grid), data.shape, margin
    )

    bbox_slices = tuple(
        slice(int(lo), int(hi) + 1)
        for lo, hi in zip(bbox[0], bbox[1], strict=True)
    )
    chunk = np.asarray(data[bbox_slices])

    world_grid = canvas_world_grid(grid)
    data_coords = plane(world_grid.reshape(-1, world_grid.shape[-1]))
    local_coords = (data_coords - bbox[0]).reshape(*grid.shape, -1)
    coords_for_map = np.moveaxis(local_coords, -1, 0)

    return ndi.map_coordinates(
        chunk, coords_for_map, order=order, cval=cval, mode='constant'
    )


def oblique_tile_to_data(
    plane: _PlaneSlice, grid: ObliqueCanvasGrid, ndim: int
) -> Affine:
    """The tile-to-data transform for a tile resampled onto `grid`.

    The resampled tile's sample (i, j) sits at world coordinate
    ``grid.world_origin + (i, j) * grid.world_step``. This returns the
    affine mapping tile sample indices to data-space coordinates such
    that composing it with the layer's (rotated) data_to_world transform
    reproduces exactly that world placement once rendered -- see the
    design doc for the derivation. Only the displayed-dims block ends up
    mattering to the renderer, which discards the rest via `set_slice`;
    the non-displayed columns are set to identity so this remains a
    well-formed tile2data transform for other consumers (e.g. coordinate
    picking).
    """
    linear_matrix = np.eye(ndim)
    linear_matrix[:, plane.displayed] = plane.matrix * grid.world_step
    translate = plane(np.asarray(grid.world_origin))
    return Affine(
        name='tile2data',
        linear_matrix=linear_matrix,
        translate=translate,
        ndim=ndim,
    )
