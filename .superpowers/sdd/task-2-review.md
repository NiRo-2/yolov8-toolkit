# Task 2 Review: YOLO26 Naming + Train Defaults

## Scope

Reviewed commit `7621c139521aa6be74622ba894f4c421fbe1f9db` against
`task-2-brief.md`, using the supplied review package. Checkout was not
modified and the full sanity suite was not rerun.

## Spec compliance: ✅

- `VRAM_PER_IMAGE` contains only `yolo26m.pt`, `yolo26l.pt`, and
  `yolo26x.pt`, with required values `0.52`, `0.58`, and `1.20`.
- Auto-selection branches, banners, help text, templates, README, and
  CLAUDE wording use YOLO / YOLO26 as required. The remaining runtime
  `YOLOv8` occurrence is the required historical FLOPs-source comment.
- The external `scripts/yolov8_pt_to_xanylabeling_onnx/` path remains
  unchanged, and README examples now use `yolov-toolkit`.
- The commit changes only the requested public tracked files; no probe or
  tiling implementation is included.
- Focused required sanity command passed:
  `select_model_and_imgsz(2000, 16)` returned `('yolo26l.pt', 1024)`;
  the printed fallback table matched the required keys and values.

## Task quality: Changes needed — Minor

1. `train_detector/train_detector.py:276` says `yolo26x` at 1280 “fits at
   batch 8 with 16GB.” With the newly required `1.20` GB/image estimate,
   the calculation is `16 * 0.85 / (1.20 * (1280 / 1024)^2) = 7.25`;
   `calc_max_batch_for_imgsz()` therefore returns 4, below `MIN_BATCH = 8`.
   The actual code correctly selects `yolo26x.pt` at 1024 in this case, so
   the decision-table comment is inaccurate. Update the row to describe
   the 1024 fallback (or revise the selection rule if 1280 is intended).

`git diff --check` passed. No Critical or Important findings.
