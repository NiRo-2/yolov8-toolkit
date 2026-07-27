# Task 5 Report: `tile_yolo_dataset` train-prep CLI

## Status

Completed and committed.

## Deliverables

- `tile_yolo_dataset/tile_yolo_dataset.py`
  - Accepts a dataset directory or `data.yaml` path.
  - Tiles each present `train`/`val`/`test` split using `tile_geometry`.
  - Converts, clips, and filters YOLO boxes; samples empty tiles deterministically.
  - Writes tiled images, labels, relative output `data.yaml`, and optional provenance manifest.
  - Skips corrupt images or labels with `[WARNING]` and auto-versions non-empty outputs.
- `tile_yolo_dataset/_Run_tile_yolo_dataset_template.bat`

## Commit

`0fa2605 Add tile_yolo_dataset train-prep CLI for imgsz windows.`

## Verification

- Synthetic 2000×1024 train image with one center label, tiled at 1024 with 0.2 overlap:
  - wrote 3 train tiles;
  - output `data.yaml` exists;
  - 3 generated label files were non-empty;
  - optional manifest contained one record per output image.
- `python -m pytest tests/test_tile_geometry.py -q` — 6 passed.
- `python -m py_compile tile_yolo_dataset/tile_yolo_dataset.py` — passed.
- IDE diagnostics — no linter errors.

## Concerns

None. `detect_images` was not modified.
