# Oblique 2D slicing for solid (shear-free) affine transforms

Tracking issue: [napari/napari#7310](https://github.com/napari/napari/issues/7310).

## Problem

When a layer's `data_to_world` transform contains a rotation (e.g. via
`layer.rotate`), switching to a 2D view slices the data with a plain
axis-aligned NumPy slice (`data[slices]`), ignoring the rotation entirely.
`_SliceInput.data_slice` (`layers/utils/_slice_input.py`) detects this case
via `is_orthogonal` and warns:

> Non-orthogonal slicing is being requested, but is not fully supported.
> Data is displayed without applying an out-of-slice rotation or shear
> component.

The data is still displayed, just wrong.

## Scope

Full generality (arbitrary shear + anisotropic scale + rotation) requires
resolving open questions about lazy-array backends (dask/zarr/tensorstore)
and anisotropic multiscale level selection. This doc scopes down to **solid
transforms**: rotation (proper or reflection) + anisotropic per-axis scale +
translation, i.e. `data_to_world.shear ≈ 0`. This covers the motivating use
case (`napari-manual-registration`, rigid-registration sliders) and is
narrow enough to ship incrementally.

The design is built so relaxing to general affines later touches a small,
isolated seam rather than a rewrite — see [Extending to general affines](#extending-to-general-affines).

### Non-goals (this doc)

- Points/Vectors/Shapes non-orthogonal slicing (separate, simpler geometry
  problem — no raster resampling involved).
- Multiscale-aware level selection under the oblique path (v1 pins to the
  layer's current `data_level`, no anisotropy-aware picking).
- Thick-slice / projection-mode combined with an oblique plane (v1 starts
  with `projection_mode == 'none'`, point-only).

## Math

A canvas pixel `(u, v)` (the two currently-displayed world dims) maps to a
world point by fixing the non-displayed world dims at `dims.point`, then to
a data point via `world_to_data`. Since `world_to_data` is affine, the
composition is affine in `(u, v)`:

```
p_data(u, v) = A @ [u, v] + b
```

- `A` = columns of `world_to_data.linear_matrix` for the displayed world
  dims — shape `(ndim, ndisplay)`.
- `b` = `world_to_data.linear_matrix @ world_point + world_to_data.translate`,
  where `world_point` has the fixed value in non-displayed dims and zero in
  displayed dims.

This holds for **any** affine, not just solid ones — the solid restriction
only gates *whether we take this path*, not the math itself.

## Pipeline

```
1. gate:      is_solid(world_to_data)                      -- solid-only check, ISOLATED
2. plane:     (A, b) from world_to_data                     -- generic, any affine
3. bbox:      map canvas-footprint corners through (A, b)   -- generic, any affine
              -> axis-aligned bbox in data space, clipped to shape
4. extract:   data_at_data_level[bbox_slices]                -- plain numpy/dask/zarr slice, unchanged
5. resample:  scipy.ndimage.map_coordinates(chunk, coords)   -- generic, any affine
              coords built from (A, b) over the canvas grid, shifted into chunk-local frame
6. tile_to_data: derived from the same (A, b), see below              -- no vispy code changes needed
```

**Step 6, corrected from an earlier draft of this doc:** `tile_to_data`
does *not* simply collapse to identity/translate — the renderer composes
`data_to_world ∘ tile_to_data` and only then restricts to the displayed
axes (`Affine.set_slice`, see `_vispy/layers/base.py:206`), so a naive
translate-only `tile_to_data` would let `data_to_world` re-apply the
rotation on top of an already-derotated image. What actually works,
reusing the exact same `plane` object from step 2 with no new math:

```python
linear_matrix = np.eye(ndim)
linear_matrix[:, displayed] = plane.matrix * world_step
translate = plane(world_origin)
```

Because `set_slice` only reads the displayed×displayed block of the
*composed* transform, and `D[displayed, :] @ D⁻¹[:, displayed] = I`
(where `D = data_to_world.linear_matrix`), this composition's displayed
block reduces to exactly `diag(world_step)` — the rotation cancels out
algebraically, it isn't just "not there". Implemented as
`oblique_tile_to_data` in `_scalar_field/_oblique_slice.py`.

**Update:** the half-pixel nuance flagged above was real and got fixed
(originally reported by a user testing the branch: after one rotation
the transform-tool bounding box was wildly mispositioned and its edges
couldn't be grabbed). Root causes, both from the same underlying
assumption — "tile pixel == data pixel" — breaking for oblique tiles:

1. **Overlay bounds.** `VispyTransformBoxOverlay`/`VispyBoundingBoxOverlay`
   are parented directly under the layer's own vispy node
   (`canvas.py:1095`), inheriting its `tile_to_data`-based local
   transform — so overlay coordinates live in *tile-pixel-index* space,
   not data space. They get their bounds from
   `_display_bounding_box_augmented_data_level`, which returns the raw
   *data* extent. For axis-aligned tiles tile-index == data-index (up to
   a known scale/translate), so this always worked; for oblique tiles
   `tile_to_data` is a rotated affine, so data-space bounds mean nothing
   in tile-index space — the box was being drawn in the wrong coordinate
   system entirely. Fixed by giving `_ScalarFieldSliceResponse` an
   `oblique_canvas_grid` field and having
   `_display_bounding_box_augmented_data_level` return tile-shape-based
   bounds when it's set (`scalar_field.py`).
2. **Pixel-center offset.** `_on_matrix_change`
   (`_vispy/layers/base.py:262-275`) computes its half-pixel shift from
   `data_to_world.set_slice(displayed).linear_matrix`, the same
   tile-pixel == data-pixel assumption. Fixed with an oblique-aware
   branch using `canvas_grid.world_step / 2` directly (uniform, since
   oblique tile pixels are a world-aligned grid) instead of routing
   through `data_to_world`.
3. **Grid origin convention** (a separate bug found while fixing #1/#2,
   in `_oblique_canvas_grid`): it used `_extent_data_augmented`
   (pixel-edge extent, `[-0.5, shape - 0.5]`) as the sampling grid's
   origin, but resample *samples* must land on pixel centers like every
   other data source in napari — the augmented extent is a derived
   quantity for drawing a box *around* a pixel-center grid, not the grid
   itself. Every sample was off by half a pixel. Fixed by switching to
   the unaugmented `_extent_data` (plus a `+1` in the sample-count
   formula, since the unaugmented extent spans `shape - 1` steps between
   `shape` samples).

Verified end-to-end: a known bright blob placed at a specific data
coordinate resamples to within 0.003 world units of its true rotated
position (`test_oblique_slice.py`), and the bounding-box overlay's
on-screen world extent matches a ground truth computed independently
from `oblique_canvas_grid` to `atol=1e-6`
(`test_vispy_bounding_box_visual.py::test_bounding_box_oblique_2D`).

Note the overlay ground truth here is the *2D slice's own footprint*,
not `layer.extent` (the full 3D data's world bounding box) — a
cross-section through a rotated volume at a fixed point generally has a
smaller/different footprint than the shadow of the whole volume. Mixing
these up cost real debugging time; worth remembering for any future
overlay work here.

Separately (pre-existing, confirmed present even for a plain unrotated
layer, **not** part of this issue): `VispyTransformBoxOverlay`'s corner
handles sit a uniform 0.5 world units inside the augmented extent
(e.g. `[-1, 29]` instead of `[-0.5, 29.5]` for a 30-cube). Small enough
not to break interaction for axis-aligned data, which is presumably why
it went unnoticed; out of scope here, worth its own issue.

## Where each piece lives

| Piece | Location | Notes |
|---|---|---|
| `is_solid(world_to_data)` | `layers/utils/_slice_input.py`, next to `is_orthogonal` | `np.allclose(world_to_data.inverse.shear, 0)` — checks the forward `data_to_world`, since inverting a shear-free transform doesn't generally stay shear-free under anisotropic scale (rotation and anisotropic scale don't commute) |
| `_PlaneSlice` type | `layers/utils/_slice_input.py` | New return type; can't reuse `_ThickNDSlice`, which *is* the orthogonal-only representation |
| `_SliceInput.slice_plane()` | `layers/utils/_slice_input.py` | Computes `(A, b)` per the math above |
| bbox + resample helpers | new module `layers/_scalar_field/_oblique_slice.py` | Kept separate from `_slice.py` control flow on purpose — this file is the only one that changes when generalizing later |
| dispatch on `_PlaneSlice` vs `_ThickNDSlice` | `_ScalarFieldSliceRequest.__call__` (`_scalar_field/_slice.py:223`) | New `_call_oblique_slice()` alongside `_call_single_scale`/`_call_multi_scale` |
| interpolation order per layer type | mirrors existing `_project_slice` override pattern (`image/_slice.py`, `labels/_slice.py`) | Image order=1 (linear), Labels order=0 (nearest — never invent label IDs) |
| multiscale level pin | `ScalarFieldSlicingState._make_slice_request_internal` (`scalar_field.py:925`) | v1: skip smart picking for oblique path, isolated as its own branch |

## Correction to initial scoping

Layer extent (`get_extent_world`, `base.py:1196`) already applies the full
`data_to_world` affine including rotation, so the world-space bounding box
used for camera fit-to-view etc. is already correct today. Only the
per-slice data *extraction* is broken — no extent changes needed.

## Extending to general affines

Two things are solid-specific, both single swappable predicates/branches:

1. `is_solid()` gate — loosen to "invertible" for the general case.
2. Multiscale level pin — replace with an anisotropy-aware picker.

Steps 2/3/5 of the pipeline (plane derivation, bbox, `map_coordinates`) are
already affine-general. Shear only makes the bbox from step 3 non-tight
(pulls more data than strictly needed) — a later optimization, not a
rewrite.

## Staged plan

1. **Math only.** `is_solid` / `_PlaneSlice` / `slice_plane` + unit tests.
   No behavior change to the existing slicing pipeline yet. **Done.**
2. Single-scale Image + Labels: wire `_call_oblique_slice` + bbox/resample
   module + `tile_to_data` derivation + Labels order=0 hook. (Labels rode
   along with this step since it was a one-line override once the
   `_project_slice`-style hook pattern was in place.) Also folded in,
   after user testing surfaced it: fixing the transform-box/bounding-box
   overlay positioning and the sampling-grid origin convention (see
   "Update" note above). **Done.**
3. Multiscale (pinned level, no smart picking).
4. Thick-slice / projection-mode interaction.

Steps 1–2 are the feasibility proof (~1–2 wk). 3–4 are incremental,
lower-risk given the seam above.

---

**Status:** Steps 1 and 2 done.

- Step 1: `is_solid`, `_PlaneSlice`, `slice_plane` in
  `layers/utils/_slice_input.py`, tests in
  `layers/utils/_tests/test_slice_input.py`.
- Step 2: new `_scalar_field/_oblique_slice.py` (canvas grid, bbox,
  `map_coordinates` resample, `tile_to_data` derivation) wired into
  `_ScalarFieldSliceRequest.__call__` (`_call_oblique_slice`) and
  `ScalarFieldSlicingState._resolve_slice_geometry`/`_oblique_canvas_grid`
  (`scalar_field.py`), gated to single-scale layers only. Labels gets a
  nearest-neighbor (`order=0`) override in `labels/_slice.py`. Tests in
  `layers/image/_tests/test_oblique_slice.py` verify actual pixel values
  against a closed-form trig ground truth (not just that code runs),
  including a case unaffected by rotation and one affected by it, plus
  regression coverage that shear and multiscale still fall back to the
  old warn-and-drop behavior. Full `layers/` suite (2067 tests) passes
  with no regressions.

Verified: the vispy node transform for a rotated 2D image comes back
well-formed (identity linear block, finite translate) rather than
degenerate — consistent with the `tile_to_data` derivation above.
Could not get an actual on-screen screenshot in this sandbox (a control
test with an ordinary unrotated image also renders blank here, so this
is a headless-environment limitation, not specific to this change) — a
real-display check is still worth doing before calling this done.

Next: step 3 (Labels order=0 hook — technically already done above,
piggybacked onto step 2), then step 4 (multiscale).
