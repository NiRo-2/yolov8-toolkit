# Task 3 Review: VRAM Probe + Cache

## Verdict

- Spec: ✅
- Quality: Approved

## Review notes

The implementation satisfies the task-scoped requirements:

- Stores the local cache at `train_detector/weights/vram_estimates.json`; that
  directory is gitignored.
- Loads valid cached estimates safely, merges writes, and falls back to the
  built-in FLOPs-scaled values (or `1.10` for unknown models).
- Uses cached estimates in both batch calculations.
- Creates a disposable synthetic 640px dataset, probes batch 2 then batch 1
  on CUDA OOM, and normalizes measured peak allocation to GB/image at 1024px.
- Runs the probe only for a missing selected model during CUDA auto-config,
  while `--probe-vram` refreshes the requested model or the yolo26m/l/x set.
- Cleanly skips the probe when CUDA is unavailable or `--device cpu` is used.
- Does not add tiling or other out-of-scope behavior.

## Verification

- `python train_detector/train_detector.py --help` exited 0 and includes
  `--probe-vram`.
- `python train_detector/train_detector.py --probe-vram --device cpu` exited
  0 and printed the CUDA-unavailable fallback message without requiring
  `--input`.
- `git diff --check 7621c139521aa6be74622ba894f4c421fbe1f9db
  30d6c4bbd09d6642ec30e55fe2105e4bcfd3829a` exited 0.

A live CUDA measurement was not required for this review and was not available;
the CPU skip path is correct.
