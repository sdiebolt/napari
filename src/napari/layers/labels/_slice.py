import numpy.typing as npt

from napari.layers._scalar_field._slice import _ScalarFieldSliceRequest
from napari.layers.base._base_constants import BaseProjectionMode
from napari.types import ArrayLike


class _LabelsSliceRequest(_ScalarFieldSliceRequest):
    @staticmethod
    def _project_slice(
        data: ArrayLike, axis: tuple[int, ...], mode: BaseProjectionMode
    ) -> npt.NDArray:
        """Project a thick slice along axis based on mode."""
        raise NotImplementedError

    @staticmethod
    def _resample_order() -> int:
        # Nearest-neighbor: linear interpolation between label IDs would
        # invent labels that were never in the data.
        return 0
