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
6. tile_to_data: identity/translate only                     -- rotation already baked in by step 5
```

Step 6 matters: `map_coordinates` is evaluated *at* the canvas sample
locations, so the output is already registered 1:1 to canvas pixels.
`tile_to_data` (consumed by vispy at `scalar_field.py:1000`) collapses to a
plain translate — no vispy-side changes needed.

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
   No behavior change to the existing slicing pipeline yet.
2. Single-scale Image, `projection_mode='none'`: wire `_call_oblique_slice`
   + bbox/resample module.
3. Labels (order=0 hook).
4. Multiscale (pinned level, no smart picking).
5. Thick-slice / projection-mode interaction.

Steps 1–2 are the feasibility proof (~1–2 wk). 3–5 are incremental,
lower-risk given the seam above.

---

**Status:** Step 1 done — `is_solid`, `_PlaneSlice`, `slice_plane` added to
`layers/utils/_slice_input.py`, with tests in
`layers/utils/_tests/test_slice_input.py`. Not yet wired into the slicing
pipeline (`_ScalarFieldSliceRequest`) — that's step 2.
