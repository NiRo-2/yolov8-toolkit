# Task 3 Report: VRAM Probe + Cache

## Status

DONE

## Scope completed

- Added a persistent, local `train_detector/weights/vram_estimates.json` cache,
  including load, merge/save, and built-in-fallback lookup helpers.
- Added a synthetic one-image temporary dataset probe that measures a single
  640px training step, retries batch sizes 2 then 1 on CUDA OOM, and scales the
  result to GB/image at 1024px.
- Added CUDA availability handling and cache-or-probe orchestration. Automatic
  configuration probes only a missing selected model before calculating batch
  and augmentation settings.
- Added `--probe-vram`, which refreshes the specified `--model` or all
  yolo26m/l/x models and exits cleanly without `--input`.
- Updated both batch calculations to retrieve cached estimates first.
- Did not implement tiling.

## Verification

- `python train_detector/train_detector.py --help` exited 0 and lists
  `--probe-vram`.
- `python train_detector/train_detector.py --probe-vram --device cpu` exited 0
  and printed the CUDA-unavailable skip message without requiring `--input`.
- Cache helper checks covered absent cache, save/load, cache precedence,
  unknown-model fallback, corrupt-cache recovery, and temporary probe dataset
  creation.
- Probe orchestration check covered forced measurement and cache persistence
  using injected CUDA/probe seams.
- Batch consumer checks confirmed the auto batch helpers execute through the
  cache-aware estimate getter.
- `python -m py_compile train_detector/train_detector.py`, `git diff --check`,
  and IDE lint diagnostics completed without errors.

## Self-review

- Confirmed the cache path is derived from the existing `WEIGHTS_DIR`.
- Confirmed the probe uses a generated temporary dataset rather than coco8.
- Confirmed only `train_detector/train_detector.py` is staged for the commit.

## Commit

`30d6c4b Add local VRAM probe cache for YOLO26 batch auto-config.`

## Concerns

- A live CUDA probe was not run because this environment reports CUDA
  unavailable. The required CPU skip path was verified.
