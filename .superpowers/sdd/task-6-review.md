# Task 6 Re-Review: Default tiled inference in `detect_images`

**Base:** `0fa2605` → **Head:** `4c34a4c` ("Pass imgsz=tile and batch tile crops in detect_image_tiled.", on top of `cd30107`)
**Scope reviewed:** `detect_images/detect_images.py` (full task range, base→head) + `.superpowers/sdd/task-6-report.md`. Cross-checked against `tile_yolo_dataset/tile_geometry.py` (unmodified, signatures confirmed) and the prior review's two blocking findings.

## Verdict: ✅ Quality Approved

Both previously-blocking findings are fixed correctly and verified directly against the current file (not just the fix commit's diff in isolation — re-read `detect_image_tiled` and its call site end-to-end).

---

## Fix verification

### 1. [Previously Critical/High] `imgsz=tile` now passed — ✅ fixed

`detect_image_tiled` (`detect_images/detect_images.py:284-309`) now calls:

```python
tile_results.extend(model(chunk, conf=conf, imgsz=tile, verbose=False))
```

`imgsz=tile` is passed explicitly on every `model()` invocation in the tiled path, so Ultralytics runs each tile crop at the tile's native size instead of falling back to the framework default of 640. This directly restores the resolution-preservation purpose of tiling for any `--tile-imgsz` value, including the common >640 case. Edge-flush tiles smaller than `tile` in one dimension (per `iter_tile_windows`'s doc comment) will still be letterboxed up to `imgsz=tile`, which is the correct behavior. Confirmed against the report's fake-model smoke output (`imgsz=tile passed explicitly to model() -> {'conf': 0.25, 'imgsz': 1024, 'verbose': False}`).

### 2. [Previously Important/Medium] Tile crops now chunked by `batch_size` — ✅ fixed

`detect_image_tiled` gained a `batch_size` parameter (default `1`, but the real call site always passes an explicit value) and chunks crops before calling the model:

```python
for chunk_start in range(0, len(crops), max(1, batch_size)):
    chunk = crops[chunk_start:chunk_start + max(1, batch_size)]
    tile_results.extend(model(chunk, conf=conf, imgsz=tile, verbose=False))
```

The call site in `main()`'s batch loop (`detect_images/detect_images.py:742-753`) passes the already-resolved `batch_size` from `resolve_batch(args.batch)` — the same value that caps the non-tiled path's per-call image count — so `--batch` now behaves consistently and meaningfully whether tiling is on or off. Per-image tile-crop count is no longer unbounded per GPU call; a large orthomosaic with hundreds of tiles will be sent in `batch_size`-sized groups instead of one giant call, addressing the OOM-risk concern from the prior review. Cross-image batching is still correctly excluded — the per-image loop at line 745 (`for idx, p, im, flat_name, display_name in valid: dets = detect_image_tiled(...)`) calls `detect_image_tiled` once per source image, so chunking is scoped to that single image's own tiles only. Confirmed against the report's smoke output (`150 tiles / batch_size=4 -> 38 model() calls (expected 38), each call carries imgsz=256`, plus a separate check that per-image call groups stay independent across two images).

Argument order at the call site (`model, im, args.conf, args.tile_imgsz, args.tile_overlap, args.tile_iou, batch_size`) matches the updated signature `(model, image_bgr, conf, tile, overlap, iou, batch_size=1)` positionally — verified directly, no mismatch.

### Bonus (not requested, non-blocking, welcome): `--tile-overlap`/`--tile-imgsz` input validation

The fix also added friendly `[ERROR]` + `sys.exit(1)` validation for `--tile-overlap` (`[0.0, 1.0)`) and `--tile-imgsz` (`> 0`) when `--tiles` is on (`detect_images/detect_images.py:492-499`), addressing prior review finding #3 (Low) as a side effect. Not required for this re-review's pass/fail but improves consistency with the rest of the script's validation style.

---

## Spec compliance (re-verified against current file)

| Constraint | Status | Notes |
|---|---|---|
| Tiles default ON | ✅ | `parser.set_defaults(tiles=True)` |
| `--no-tiles` restores whole-image inference | ✅ | unchanged whole-image branch, `results_to_dets(r)` adapter used |
| `--tile-imgsz` default 1024, now actually controls inference resolution | ✅ | `imgsz=tile` passed to `model()` — this is the fix |
| `--tile-overlap` default 0.2 | ✅ | validated range `[0.0, 1.0)` |
| `--tile-iou` default 0.5, passed to `nms_xyxy(all_dets, iou_thresh=iou)` | ✅ | unchanged |
| Tile crops batched by `--batch`, not sent as one unbounded call | ✅ | this is the fix — chunked via `range(0, len(crops), max(1, batch_size))` |
| Tiles from different source images never batched together | ✅ | per-image loop calls `detect_image_tiled` once per image; chunking is scoped inside that call only |
| Map boxes back to full-image space | ✅ | `results_to_dets(result, x_offset=tx1, y_offset=ty1)`, offsets applied per chunked-tile result correctly zipped against `windows` |
| NMS merge across all of one image's tiles (not per-chunk) | ✅ | `all_dets` accumulates across *all* chunks before the single `nms_xyxy(all_dets, iou_thresh=iou)` call — chunking the model calls does not fragment the NMS merge |
| `iter_tile_windows`/`nms_xyxy` call signatures match `tile_geometry.py` | ✅ | `iter_tile_windows(width, height, tile, overlap)` and `nms_xyxy(dets, iou_thresh=0.5)` — argument order confirmed against source |

All items ✅. No regressions introduced by the fix commit into the surrounding whole-image path, JSON export, drawing, or worker `post_process` logic (all unchanged from the already-approved structural parts of the original review).

---

## Residual non-blocking notes (informational only, not required for approval)

- Batch-size default of `1` on `detect_image_tiled`'s signature is dead in practice (the only call site always passes the resolved `batch_size`); harmless, just noting it's not independently reachable.
- No committed automated regression test for the tiled path (same Low/informational note as the original review — a `FakeModel`-style unit test would still be valuable but is not a blocking requirement).
- Not re-validated against real `.pt` weights (no model available in this task's scope, consistent with prior report and review).

## Summary

Both blocking findings from the first review — missing `imgsz=tile` and unbounded per-image tile batching — are fixed correctly, verified against the live file (not just the diff), including argument-order and NMS-scope correctness across the new chunking loop. Spec compliance: all items ✅.

**Quality Approved.**
