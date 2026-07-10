from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import pint

from napari.utils.misc import reorder_after_dim_reduction
from napari.utils.transforms import Affine
from napari.utils.translations import trans

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Self

    import numpy.typing as npt

    from napari.components.dims import Dims

_T = TypeVar('_T')


@dataclass(frozen=True)
class _ThickNDSlice(Generic[_T]):
    """Holds the point and the left and right margins of a thick nD slice."""

    point: tuple[_T, ...]
    margin_left: tuple[_T, ...]
    margin_right: tuple[_T, ...]

    @property
    def ndim(self):
        return len(self.point)

    @classmethod
    def make_full(
        cls,
        point=None,
        margin_left=None,
        margin_right=None,
        ndim=None,
    ):
        """
        Make a full slice based on minimal input.

        If ndim is provided, it will be used to crop or prepend zeros to the given values.
        Values not provided will be filled zeros.
        """
        for val in (point, margin_left, margin_right):
            if val is not None:
                val_ndim = len(val)
                break
        else:
            if ndim is None:
                raise ValueError(
                    'ndim must be provided if no other value is given'
                )
            val_ndim = ndim

        ndim = val_ndim if ndim is None else ndim

        # not provided arguments are just all zeros
        point = (0,) * ndim if point is None else tuple(point)
        margin_left = (
            (0,) * ndim if margin_left is None else tuple(margin_left)
        )
        margin_right = (
            (0,) * ndim if margin_right is None else tuple(margin_right)
        )

        # prepend zeros if ndim is bigger than the given values
        prepend = max(ndim - val_ndim, 0)

        point = (0,) * prepend + point
        margin_left = (0,) * prepend + margin_left
        margin_right = (0,) * prepend + margin_right

        # crop to ndim in case given values are longer (keeping last dims)
        return cls(
            point=point[-ndim:],
            margin_left=margin_left[-ndim:],
            margin_right=margin_right[-ndim:],
        )

    @classmethod
    def from_dims(cls, dims: Dims) -> Self:
        """Generate from a Dims object's point and margins."""
        return cls.make_full(dims.point, dims.margin_left, dims.margin_right)

    def copy_with(
        self,
        point=None,
        margin_left=None,
        margin_right=None,
        ndim=None,
    ):
        """Create a copy, but modifying the given fields."""
        return self.make_full(
            point=point or self.point,
            margin_left=margin_left or self.margin_left,
            margin_right=margin_right or self.margin_right,
            ndim=ndim or self.ndim,
        )

    def as_array(self) -> npt.NDArray:
        """Return point and left and right margin as a (3, D) array."""
        return np.array([self.point, self.margin_left, self.margin_right])

    @classmethod
    def from_array(cls, arr: npt.NDArray) -> _ThickNDSlice:
        """Construct from a (3, D) array of point, left margin and right margin."""
        return cls(
            point=tuple(arr[0]),
            margin_left=tuple(arr[1]),
            margin_right=tuple(arr[2]),
        )

    def __getitem__(self, key):
        # this allows to use numpy-like slicing on the whole object
        return _ThickNDSlice(
            point=tuple(np.array(self.point)[key]),
            margin_left=tuple(np.array(self.margin_left)[key]),
            margin_right=tuple(np.array(self.margin_right)[key]),
        )

    def __iter__(self):
        # iterate all three fields dimension per dimension
        yield from zip(
            self.point, self.margin_left, self.margin_right, strict=False
        )


@dataclass(frozen=True)
class _PlaneSlice:
    """An oblique (non-axis-aligned) data-space slicing plane.

    Represents the affine map from canvas pixel coordinates (the layer's
    displayed world dims, in world units) to full data-space coordinates::

        data_coord = matrix @ canvas_coord + offset

    This is the non-orthogonal counterpart of `_ThickNDSlice`: where
    `_ThickNDSlice` can only represent an axis-aligned point and margins
    per non-displayed dimension, `_PlaneSlice` represents an arbitrary
    (solid, i.e. shear-free) plane through the data.

    Attributes
    ----------
    matrix : (ndim, ndisplay) array
        Maps a displayed-dim canvas coordinate to a data-space offset.
    offset : (ndim,) array
        The data-space coordinate at the canvas origin.
    displayed : tuple of int
        The layer dimension indices displayed in this slice, matching
        `_SliceInput.displayed`.
    not_displayed : tuple of int
        The layer dimension indices not displayed in this slice, matching
        `_SliceInput.not_displayed`.
    """

    matrix: npt.NDArray
    offset: npt.NDArray
    displayed: tuple[int, ...]
    not_displayed: tuple[int, ...]

    @property
    def ndim(self) -> int:
        """The dimensionality of the full data-space coordinates."""
        return self.offset.shape[0]

    def __call__(self, canvas_coords: npt.NDArray) -> npt.NDArray:
        """Maps canvas coordinates with shape (..., ndisplay) to data coordinates with shape (..., ndim)."""
        canvas_coords = np.asarray(canvas_coords)
        return canvas_coords @ self.matrix.T + self.offset


@dataclass(frozen=True)
class _SliceInput:
    """Encapsulates the input needed for slicing a layer.

    An instance of this should be associated with a layer and some of the values
    in ``Viewer.dims`` when slicing a layer.
    """

    # The number of dimensions to be displayed in the slice.
    ndisplay: int
    # The thick slice in world coordinates.
    # Only the elements in the non-displayed dimensions have meaningful values.
    world_slice: _ThickNDSlice[float]
    # The layer dimension indices in the order they are displayed.
    # A permutation of the ``range(self.ndim)``.
    # The last ``self.ndisplay`` dimensions are displayed in the canvas.
    order: tuple[int, ...]

    @property
    def ndim(self) -> int:
        """The dimensionality of the associated layer."""
        return len(self.order)

    @property
    def displayed(self) -> list[int]:
        """The layer dimension indices displayed in this slice."""
        return list(self.order[-self.ndisplay :])

    @property
    def not_displayed(self) -> list[int]:
        """The layer dimension indices not displayed in this slice."""
        return list(self.order[: -self.ndisplay])

    def with_ndim(self, ndim: int) -> _SliceInput:
        """Returns a new instance with the given number of layer dimensions."""
        old_ndim = self.ndim
        world_slice = self.world_slice.copy_with(ndim=ndim)
        if old_ndim > ndim:
            order = reorder_after_dim_reduction(self.order[-ndim:])
        elif old_ndim < ndim:
            order = tuple(range(ndim - old_ndim)) + tuple(
                o + ndim - old_ndim for o in self.order
            )
        else:
            order = self.order

        return _SliceInput(
            ndisplay=self.ndisplay, world_slice=world_slice, order=order
        )

    def data_slice(
        self,
        world_to_data: Affine,
    ) -> _ThickNDSlice[float | int]:
        """Transforms this thick_slice into data coordinates with only relevant dimensions.

        The elements in non-displayed dimensions will be real numbers.
        The elements in displayed dimensions will be ``slice(None)``.
        """
        if not self.is_orthogonal(world_to_data):
            warnings.warn(
                trans._(
                    'Non-orthogonal slicing is being requested, but is not fully supported. '
                    'Data is displayed without applying an out-of-slice rotation or shear component.',
                    deferred=True,
                ),
                category=UserWarning,
            )

        slice_world_to_data = world_to_data.set_slice(self.not_displayed)
        world_slice_not_disp = self.world_slice[self.not_displayed].as_array()

        data_slice = slice_world_to_data(world_slice_not_disp)
        # the margins (data_slice[1:]) should be relative to the origin,
        # but properly rescaled based on the transform, so we remove the translation
        # to bring them back around zero.
        data_slice[1:] -= slice_world_to_data.translate

        full_data_slice = np.full((3, self.ndim), np.nan)

        for i, ax in enumerate(self.not_displayed):
            # we cannot have nan in non-displayed dims, so we default to 0
            full_data_slice[:, ax] = np.nan_to_num(data_slice[:, i], nan=0)

        return _ThickNDSlice.from_array(full_data_slice)

    def is_orthogonal(self, world_to_data: Affine) -> bool:
        """Returns True if this slice represents an orthogonal slice through a layer's data, False otherwise."""
        # Subspace spanned by non displayed dimensions
        non_displayed_subspace = np.zeros(self.ndim)
        for d in self.not_displayed:
            non_displayed_subspace[d] = 1
        # Map subspace through inverse transform, ignoring translation
        world_to_data = Affine(
            ndim=self.ndim,
            linear_matrix=world_to_data.linear_matrix,
            translate=None,
        )
        mapped_nd_subspace = world_to_data(non_displayed_subspace)
        # Look at displayed subspace
        displayed_mapped_subspace = (
            mapped_nd_subspace[d] for d in self.displayed
        )
        # Check that displayed subspace is null
        return all(abs(v) < 1e-8 for v in displayed_mapped_subspace)

    def is_solid(self, world_to_data: Affine) -> bool:
        """Returns True if world_to_data is the inverse of a solid (shear-free) transform.

        A solid transform is a composition of a rotation (or reflection),
        an anisotropic per-axis scale, and a translation, with no shear --
        i.e. a `layer.data_to_world` set only via `rotate`/`scale`/
        `translate`. Every orthogonal slice (see `is_orthogonal`) is also
        solid, since axis-aligned slicing only requires the non-displayed
        subspace to map cleanly, whereas solid-ness is a property of the
        whole transform.

        Note this checks `world_to_data.inverse` (i.e. `data_to_world`),
        not `world_to_data` itself: inverting a shear-free transform does
        not generally stay shear-free when the scale is anisotropic,
        because rotation and anisotropic scale don't commute.

        Oblique (non-orthogonal) slicing through a solid transform can be
        implemented by resampling along the plane spanned by the displayed
        dims (see `slice_plane`), without needing to handle shear.
        """
        return bool(np.allclose(world_to_data.inverse.shear, 0, atol=1e-8))

    def slice_plane(self, world_to_data: Affine) -> _PlaneSlice:
        """Computes the data-space plane sampled by this oblique slice.

        This should only be used when `is_solid(world_to_data)` is True.
        It returns the affine map from canvas pixel coordinates (in the
        displayed dims, in world units) to full data-space coordinates:

            data_coord = matrix @ canvas_coord + offset

        Unlike `data_slice`, this does not drop the out-of-slice rotation
        component, so it can be used to correctly resample an oblique
        (non-axis-aligned) slice through the data.
        """
        linear_matrix = world_to_data.linear_matrix
        matrix = linear_matrix[:, self.displayed]

        world_point = np.zeros(self.ndim)
        for d in self.not_displayed:
            world_point[d] = self.world_slice.point[d]
        offset = linear_matrix @ world_point + world_to_data.translate

        return _PlaneSlice(
            matrix=matrix,
            offset=offset,
            displayed=tuple(self.displayed),
            not_displayed=tuple(self.not_displayed),
        )


def apply_units_to_transform(
    data_to_world: Affine, world_units: Sequence[str] | None
) -> Affine:
    """Applies unit scaling to a data_to_world transform.

    Parameters
    ----------
    data_to_world : Affine
        The original data to world transform.
    world_units : Sequence[str] | None
        The units for each dimension of the layer.

    Returns
    -------
    Affine
        The new data to world transform with unit scaling applied.
    """
    if world_units is None:
        return data_to_world

    layer_units = data_to_world.units
    if len(world_units) < len(layer_units):
        return data_to_world

    reg = pint.get_application_registry()
    scale = tuple(
        reg.get_base_units(x)[0] / reg.get_base_units(y)[0]
        for x, y in zip(
            world_units[-len(layer_units) :], layer_units, strict=True
        )
    )
    scale_matrix = np.diag(scale)
    new_linear = data_to_world.linear_matrix @ scale_matrix
    return Affine(
        ndim=data_to_world.ndim,
        linear_matrix=new_linear,
        translate=data_to_world.translate,
    )
