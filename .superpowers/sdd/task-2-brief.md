### Task 2: YOLO26 naming + train defaults (no probe yet)

**Files:**
- Modify: `train_detector/train_detector.py` (VRAM table, `select_model_and_imgsz`, help text, headers)
- Modify: all other `*.py` / `*_template.bat` / `README.md` / `CLAUDE.md` user-facing “YOLOv8” / `yolov8*.pt` strings (examples + banners). Keep external X-AnyLabel script path `scripts/yolov8_pt_to_xanylabeling_onnx/` unchanged.
- Fix: README example path `yolov8-toolkit` → `yolov-toolkit`

**Interfaces:**
- Consumes: none
- Produces: `VRAM_PER_IMAGE` keys `yolo26m.pt`/`yolo26l.pt`/`yolo26x.pt` with values `0.52`/`0.58`/`1.20`; auto-select returns those filenames

- [ ] **Step 1: Replace train_detector VRAM + selection**

In `train_detector/train_detector.py`, set:

```python
# VRAM usage estimates per image at 1024px (GB).
# FLOPs-scaled from prior YOLOv8 measurements using Ultralytics published
# detect FLOPs (m 68.2/78.9, l 86.4/165.2, x 193.9/257.8). Local probe cache
# overrides these when present (see load_vram_estimates / probe_vram).
VRAM_PER_IMAGE = {
    "yolo26m.pt": 0.52,
    "yolo26l.pt": 0.58,
    "yolo26x.pt": 1.20,
}
```

Update `select_model_and_imgsz` decision table comments and every `"yolov8m.pt"` / `"yolov8l.pt"` / `"yolov8x.pt"` string to the matching `yolo26*.pt`. Update module docstring, argparse description/help (`e.g. yolo26x.pt`), and `snap_to_standard` docstring (“standard YOLO imgsz”).

- [ ] **Step 2: Sweep remaining public strings**

Replace user-facing `YOLOv8` → `YOLO` (or `YOLO26` only where a concrete model is meant). Replace example weights `yolov8m.pt` → `yolo26m.pt` (and l/x). README H1: `# YOLO Toolkit` with one-line “latest Ultralytics YOLO (currently YOLO26)”. Do **not** rename the external path `yolov8_pt_to_xanylabeling_onnx`.

- [ ] **Step 3: Sanity check defaults**

Run:

```bash
python -c "from train_detector.train_detector import select_model_and_imgsz, VRAM_PER_IMAGE; print(select_model_and_imgsz(2000, 16)); print(VRAM_PER_IMAGE)"
```

Expected: `('yolo26l.pt', 1024)` (or whatever the table yields for 2000 imgs / 16GB) and the three yolo26 keys.

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "Retarget toolkit branding and train defaults to YOLO26."
```

(Only stage tracked public files; do not add personal bats/weights.)

---

