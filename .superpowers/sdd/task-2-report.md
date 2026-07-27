# Task 2 Report: YOLO26 Naming + Train Defaults

## Status

DONE

## Scope completed

- Replaced the training fallback table with `yolo26m.pt`, `yolo26l.pt`, and
  `yolo26x.pt` at `0.52`, `0.58`, and `1.20` GB per image at 1024px.
- Updated every auto-selection branch, its decision table, training help text,
  and the standard-image-size docstring to use the new naming.
- Rebranded public script banners, help text, templates, README, and CLAUDE
  wording from YOLOv8 to YOLO or YOLO26 where a concrete model is referenced.
- Updated README examples from `yolov8*.pt` to `yolo26*.pt` and corrected the
  repository example path to `yolov-toolkit`.
- Preserved every external X-AnyLabel path under
  `scripts/yolov8_pt_to_xanylabeling_onnx/`.
- Did not implement VRAM probing or tiling.

## Verification

Ran the required sanity command:

```text
python -c "from train_detector.train_detector import select_model_and_imgsz, VRAM_PER_IMAGE; print(select_model_and_imgsz(2000, 16)); print(VRAM_PER_IMAGE)"
('yolo26l.pt', 1024)
{'yolo26m.pt': 0.52, 'yolo26l.pt': 0.58, 'yolo26x.pt': 1.2}
```

Also ran `python -m py_compile` over all modified Python scripts, `git diff
--check`, and IDE lint checks for the modified Python files. All completed
without errors.

## Self-review

- Confirmed every targeted user-facing YOLOv8 model-name example and branding
  string was updated; the only retained targeted-code `YOLOv8` reference is
  the required FLOPs-source comment in `train_detector.py`.
- Confirmed the required external X-AnyLabel path remains unchanged.
- Confirmed the commit stages only tracked public files; `.superpowers/`
  remained untracked.

## Commit

`7621c13 Retarget toolkit branding and train defaults to YOLO26.`

## Concerns

None.
