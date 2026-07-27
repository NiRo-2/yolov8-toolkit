# Task 6 Report: Default tiled inference in `detect_images`

## Status: Complete

## Changes
- Imported `iter_tile_windows`, `nms_xyxy` from `tile_yolo_dataset/tile_geometry.py` (sys.path insert, same pattern as `ortho_tag_sidecar`).
- Added CLI flags: `--tiles`/`--no-tiles` (default tiles on), `--tile-imgsz` (1024), `--tile-overlap` (0.2), `--tile-iou` (0.5). Updated module docstring and `[Config]` print block.
- Refactored `draw_detections` and `export_json` to accept a flat `list[{"cls","conf","x1","y1","x2","y2"}]` instead of ultralytics `Results`.
- Added `results_to_dets()` adapter (ultralytics `Results` → flat dets, with optional x/y offset) and `detect_image_tiled(model, image_bgr, conf, tile, overlap, iou)`, which crops all tile windows for one source image, runs them as a single batched `model()` call, offsets coords back to source-image space, and merges with `nms_xyxy` (skipped when only one window, i.e. image fits in one tile).
- Batch loop: when `--tiles` (default), each source image is tiled+detected individually via `detect_image_tiled` (tiles from different source images are never batched together, per source image's own tiles are batched in one call); when `--no-tiles`, unchanged whole-image batching via `results_to_dets`. `ThreadPoolExecutor` post-processing (`post_process`) unchanged except it now takes `dets` instead of `results`.

## Commits
- `cd30107` "Enable default tiled detection with full-image NMS merge." (`detect_images/detect_images.py` only)

## Test summary
- `--help` verified: shows `--tiles/--no-tiles`, `--tile-imgsz`, `--tile-overlap`, `--tile-iou` with stated defaults.
- Ran a standalone smoke script with a `FakeModel` (no real weights needed): confirmed (1) small image → single window, single `model()` call, correct absolute coords; (2) large image (2000x3000, tile 1024/overlap 0.2) → 12 tiles batched into **one** `model()` call, all merged det coords in-bounds; (3) `results_to_dets` adapter produces expected dict; (4) refactored `draw_detections`/`export_json` work correctly on flat det lists.
- No linter errors (`ReadLints`).

## Concerns
- Did not run against a real `.pt` model / real images (none available in this task scope) — only structural/logic smoke tests via a fake model stand-in.
- NMS merge quality at tile seams (double-detections split across tile boundaries) not empirically validated with real weights; relies on `nms_xyxy` correctness (already unit-tested in `tests/test_tile_geometry.py` per Task brief context).

Report path: `d:\Nir\DevProjects\yolov-toolkit\.superpowers\sdd\task-6-report.md`

## Fix: Critical/Important review findings

### Changes
- `detect_image_tiled` now passes `imgsz=tile` explicitly to every `model(...)` call, so Ultralytics runs tile crops at the tile's native size instead of silently resizing to its default 640.
- `detect_image_tiled` gained a `batch_size` parameter (default `1`) and now chunks a single image's tile crops into groups of at most `batch_size` before each `model()` call, instead of one giant call with all tiles. The call site in `main()` passes the already-resolved `batch_size` (from `resolve_batch(args.batch)`). Tiles from different source images are still never batched together (per-image loop unchanged).
- Added friendly CLI validation (only when `--tiles` is on): `--tile-overlap` must be in `[0.0, 1.0)` and `--tile-imgsz` must be `> 0`, each failing with `[ERROR] ...` + `sys.exit(1)` instead of letting `iter_tile_windows` raise a raw `ValueError`.

### Commits
- `db690c8` "Pass imgsz=tile and batch tile crops in detect_image_tiled." (`detect_images/detect_images.py` only)

### Test commands / output
`--help` (flags unchanged, still lists `--tiles/--no-tiles`, `--tile-imgsz`, `--tile-overlap`, `--tile-iou`):
```
python detect_images/detect_images.py --help
```

Standalone fake-model smoke script (`_tmp_task6_fix_smoke.py`, written temporarily, run, then deleted — not committed):
```
$ python _tmp_task6_fix_smoke.py
PASS: imgsz=tile passed explicitly to model() -> {'conf': 0.25, 'imgsz': 1024, 'verbose': False}
PASS: 150 tiles / batch_size=4 -> 38 model() calls (expected 38), each call carries imgsz=256
PASS: per-image call groups stay separate (38 calls per image, no cross-image batching)

All smoke tests passed.
```
Checks performed: (1) a small image (single tile window) triggers exactly one `model()` call with `imgsz=1024` in kwargs; (2) a large image (2000x3000, tile=256, overlap=0.2 → 150 tile windows) with `batch_size=4` triggers exactly `ceil(150/4)=38` `model()` calls, each with `imgsz=256`, no call exceeding 4 crops, and total crops across calls == 150; (3) calling `detect_image_tiled` for two separate images in sequence produces two independent, equal-sized groups of calls (no cross-image batching).

Validation smoke test (manual, ad-hoc, model path stubbed to bypass the earlier "model not found" check):
```
$ python detect_images/detect_images.py --images . --model _tmp_fake_model.pt --tile-overlap 1.5
[ERROR] --tile-overlap must be in [0.0, 1.0), got: 1.5
$ python detect_images/detect_images.py --images . --model _tmp_fake_model.pt --tile-imgsz 0
[ERROR] --tile-imgsz must be > 0, got: 0
$ python detect_images/detect_images.py --images . --model _tmp_fake_model.pt --tile-overlap -0.1
[ERROR] --tile-overlap must be in [0.0, 1.0), got: -0.1
```

`ReadLints` on `detect_images/detect_images.py`: no linter errors.

### Concerns
- Not re-validated against real `.pt` weights (none available in this scope); relies on the fake-model smoke harness for call-shape/kwargs assertions, same limitation as the original task.
