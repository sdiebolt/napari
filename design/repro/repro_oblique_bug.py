# Reproduction script for the two-rotation oblique-slicing bug (#7310).
#
# STATUS: the bug this was written to chase (transform-box widget
# mispositioned after a rotation) is fixed -- see the "Update" note in
# design/7310-solid-affine-oblique-slicing.md for the root causes. This
# script still works as a manual regression check: run cells [1]-[4],
# then [5a] or [5b], and the transform box (cell [5b]) should now track
# the rotated layer correctly.
#
# Run cell by cell (VS Code "Run Cell" / Jupyter interactive window both
# understand the `# %%` markers below). Or paste each cell into napari's
# built-in Python console one at a time.

# %% [1] Setup: make errors during async slicing print a full traceback to
# the terminal instead of just a GUI toast notification, so we can actually
# see what breaks.
import numpy as np
import napari
from napari.utils.notifications import notification_manager, NotificationSeverity


def _print_full_traceback(notification):
    if notification.severity == NotificationSeverity.ERROR:
        import traceback

        print('=' * 70)
        print('NAPARI ERROR NOTIFICATION:')
        traceback.print_exception(
            type(notification.exception),
            notification.exception,
            notification.exception.__traceback__,
        )
        print('=' * 70)


notification_manager.notification_ready.connect(_print_full_traceback)


def rotation_in_axes(ndim, axes, angle_degrees):
    """A rotation matrix rotating only within the given pair of axes."""
    i, j = axes
    m = np.eye(ndim)
    c, s = np.cos(np.radians(angle_degrees)), np.sin(np.radians(angle_degrees))
    m[i, i] = c
    m[i, j] = -s
    m[j, i] = s
    m[j, j] = c
    return m


# %% [2] Synthetic 3D data with a distinct band on each axis, so rotation is
# visually obvious: bright plane = mid-Z, mid plane = mid-Y, dim plane = mid-X.
N = 80
data = np.zeros((N, N, N), dtype=np.uint8)
data[N // 2 - 2 : N // 2 + 2, :, :] = 255  # z band
data[:, N // 2 - 2 : N // 2 + 2, :] = 150  # y band
data[:, :, N // 2 - 2 : N // 2 + 2] = 80  # x band

# Change this if your real data has anisotropic voxels -- that's relevant,
# rotation + anisotropic scale is where the earlier bug (composing world_to_data
# vs data_to_world) lived.
scale = (1.0, 1.0, 1.0)

viewer = napari.Viewer()
layer = viewer.add_image(data, scale=scale, name='repro')
viewer.dims.ndisplay = 3

# %% [3] First rotation (around z-ish, axes 0-1) applied via code, then
# switch to 2D. This is the case that reportedly "worked fine" for you.
layer.rotate = rotation_in_axes(3, (0, 1), 25.0)
viewer.dims.ndisplay = 2
viewer.dims.set_point(0, N // 2)
viewer.reset_view()

print('after rotation 1:')
print('  layer.rotate =\n', layer.rotate)
print('  layer._data_to_world.shear =', layer._data_to_world.shear)

# %% [4] Diagnostic snapshot -- run this any time to inspect current state
# without changing anything.
from napari.layers.utils._slice_input import _SliceInput

si: _SliceInput = layer._slice_input
w2d = layer._data_to_world.inverse
print('slice_input.displayed =', si.displayed, ' not_displayed =', si.not_displayed)
print('is_orthogonal =', si.is_orthogonal(w2d))
print('is_solid =', si.is_solid(w2d))
print('multiscale =', layer.multiscale)
print('current slice image shape =', np.asarray(layer._slice.image.view).shape)

# %% [5a] EITHER: second rotation via code (around y-ish, axes 0-2),
# deterministic -- run this to test the pure-code path.
layer.rotate = rotation_in_axes(3, (0, 2), 20.0) @ layer.rotate
viewer.dims.ndisplay = 2
viewer.reset_view()
print('after rotation 2 (code path):')
print('  layer.rotate =\n', layer.rotate)

# re-run cell [4] now to see is_solid / is_orthogonal / any traceback above.

# %% [5b] OR: second rotation via the GUI -- do this INSTEAD of [5a] if you
# want to reproduce the interactive transform-tool breakage specifically.
# 1. Select the 'repro' layer in the layer list.
# 2. Click the transform tool icon in the layer controls (top of the panel).
# 3. Drag a corner/edge handle to rotate around a DIFFERENT axis than the
#    one used in cell [3].
# 4. Then run cell [4] again to snapshot state, and check the terminal for
#    any traceback printed by the hook installed in cell [1].

# %% [6] Force a fresh re-slice and watch for warnings/errors explicitly.
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    viewer.dims.set_point(0, N // 2 + 1)
    viewer.dims.set_point(0, N // 2)
    for w in caught:
        print('WARNING:', w.category.__name__, '-', w.message)
