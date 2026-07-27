### Task 5: `tile_yolo_dataset` train-prep CLI

**Files:**
- Create: `tile_yolo_dataset/tile_yolo_dataset.py`
- Create: `tile_yolo_dataset/_Run_tile_yolo_dataset_template.bat`
- Modify: `README.md` / `CLAUDE.md` script tables (can wait until Task 7 if preferred; include minimal docstring usage here)

**Interfaces:**
- Consumes: `tile_geometry` helpers from Task 4; `resolve_output_dir` pattern from `flat_yolo_split.py`
- Produces: CLI  
  `python tile_yolo_dataset/tile_yolo_dataset.py --input DIR --output DIR [--imgsz 1024] [--overlap 0.2] [--empty-frac 0.10] [--seed 42] [--manifest]`

- [ ] **Step 1: Implement CLI skeleton matching flat_yolo_split style**

Required behavior:

1. Load input `data.yaml` (must exist under `--input` or be `--input` file path — support both: if `--input` is a yaml file use its parent as dataset root; if directory, require `data.yaml` inside).
2. For each split present (`train`/`val`/`test`): resolve images dir; pair with sibling `labels/` (replace `/images` with `/labels` on the path).
3. For each image: read size via OpenCV; `iter_tile_windows`; for each window crop with `im[y1:y2, x1:x2]`; convert labels; clip; keep if `keep_clipped_box`.
4. Collect labelled tile records and empty tile records; `select_empty_tiles` on empties; write selected tiles as `{stem}_x{x1}_y{y1}{ext}` + matching `.txt`.
5. Copy class names/`nc` into new `data.yaml` with relative `train: train/images` etc.
6. If `--manifest`: write `tiles_manifest.json` list of `{split, source, tile_x1, tile_y1, tile_x2, tile_y2, out_name, n_labels}`.
7. Auto-version non-empty output dirs like `resolve_output_dir` in `flat_yolo_split.py`.

Skip corrupt images/labels with `[WARNING]` (do not hard-fail the whole run unless zero tiles written).

- [ ] **Step 2: Template bat**

```bat
@echo off
REM Tile a YOLO-format dataset into imgsz windows (default 1024, 20%% overlap) for small-object training.
python "%~dp0tile_yolo_dataset.py" --input "C:\path\to\dataset" --output "C:\path\to\dataset_tiled" --imgsz 1024 --overlap 0.2
pause
```

- [ ] **Step 3: Smoke test on tiny synthetic dataset**

Create temp dataset with one 2000×1024 image, one label center box, run the script, assert:

- more than one train tile written
- `data.yaml` exists
- at least one label file non-empty

```bash
python tile_yolo_dataset/tile_yolo_dataset.py --input <tmp_in> --output <tmp_out> --imgsz 1024 --overlap 0.2
```

- [ ] **Step 4: Commit**

```bash
git add tile_yolo_dataset/tile_yolo_dataset.py tile_yolo_dataset/_Run_tile_yolo_dataset_template.bat
git commit -m "Add tile_yolo_dataset train-prep CLI for imgsz windows."
```

---

