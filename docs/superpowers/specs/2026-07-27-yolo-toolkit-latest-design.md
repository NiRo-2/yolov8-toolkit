# YOLO Toolkit → latest Ultralytics YOLO (YOLO26) migration

**Date:** 2026-07-27  
**Status:** Approved design (pending implementation plan)  
**Repo:** public [`NiRo-2/yolov-toolkit`](https://github.com/NiRo-2/yolov-toolkit) (GitHub rename already done; was `yolov8-toolkit`)

## Goal

Keep this toolkit version-agnostic as **YOLO Toolkit**, always targeting the **latest Ultralytics YOLO** generation. As of this writing that is **YOLO26**. Update docs and code accordingly, recalibrate training VRAM heuristics via a local auto-probe, add **default-on image tiling** for small-object train/infer on large images, and harden gitignore so personal/cache/weights never ship in a public commit.

## Decisions (locked)

| Topic | Choice |
|---|---|
| Brand | **YOLO Toolkit** (not “YOLO26 Toolkit”) |
| Dataset wording | **YOLO format** (not “YOLOv8 format”) |
| Current generation | YOLO26 (`yolo26{n,s,m,l,x}.pt`) |
| Auto-selected sizes | Keep **m / l / x** only |
| VRAM strategy | FLOPs-scaled built-in fallbacks + local auto-probe cache |
| Probe UX | Auto on first GPU train if cache missing; `--probe-vram` refreshes |
| Approach | Rename + fallbacks + probe (approach 2) |
| Tiling scope | **Train + infer**, default on |
| Tiling architecture | New `tile_yolo_dataset/` + tiled mode in `detect_images` |
| Tile policy | Tile size = `imgsz`, overlap **20%**, skip if both sides ≤ imgsz |
| Border labels (train) | Clip to tile; drop if remaining area &lt; **20%** of original |
| Empty tiles (train) | Keep a **capped** negative set (~10% of output tiles) |
| Infer merge | Map to full-image coords + class-wise NMS IoU **0.5** |
| Opt out | `--no-tiles` on detect; skip/omit tile step only if user chooses |

## Scope

### In scope

- Rebrand docs/user-facing strings from YOLOv8 → YOLO Toolkit / YOLO format, noting current generation is YOLO26
- Update training defaults and examples to `yolo26m.pt` / `yolo26l.pt` / `yolo26x.pt`
- FLOPs-scaled built-in `VRAM_PER_IMAGE` fallbacks for YOLO26
- Auto VRAM probe + local cache for `train_detector`
- New `tile_yolo_dataset/` script (train-prep tiling) + template bat
- Default-on tiled inference + NMS merge in `detect_images` (`--no-tiles` to disable)
- Tighten `.gitignore` for public-repo safety (especially `.cursor/`)
- Align template `.bat` REM lines and `requirements.txt` commentary / minimum pin
- Point local `origin` at `https://github.com/NiRo-2/yolov-toolkit.git` if it still references the old `yolov8-toolkit` remote URL
- Fix leftover example paths that say `yolov8-toolkit` → `yolov-toolkit`

### Already done (not re-done here)

- GitHub repository rename to [`NiRo-2/yolov-toolkit`](https://github.com/NiRo-2/yolov-toolkit) (manual)

### Out of scope

- Renaming the local disk checkout folder (already matches `yolov-toolkit` on this machine; optional elsewhere)
- Changing the external [X-AnyLabel-toolkit](https://github.com/NiRo-2/X-AnyLabel-toolkit) export script (keep existing pointer; script path/name unchanged)
- Running GPU VRAM measurement in the implementation session (probe does that at runtime)
- Changing auto-defaults to still prefer `yolov8*.pt` (manual `--model yolov8m.pt` may still work if Ultralytics serves it)
- Ignoring the entire `.claude/` tree (keep existing `settings.local.json` rule only)
- Structural rewrite of the dataset-size × VRAM decision-table tiers
- Soft-NMS / weighted box fusion for tile merge
- Auto-tiling inside `train_detector` (train consumes an already-tiled YOLO dataset)
- Baking tiling into `voc_to_yolo` / `vlm_yolo_prep` / `flat_yolo_split` / `remap_yolo_labels` (those remain whole-image producers; tiling is a dedicated next step)

## Naming conventions

| Surface | Convention |
|---|---|
| README H1 / product name | **YOLO Toolkit** |
| Current-generation note | “Uses the latest Ultralytics YOLO (currently **YOLO26**)” |
| Dataset layout | **YOLO format** (`train/val/[test]/images|labels` + `data.yaml`) |
| Model filenames | `yolo26{n,s,m,l,x}.pt` (no `v` in the stem) |
| Auto-selected models | `yolo26m.pt`, `yolo26l.pt`, `yolo26x.pt` |
| CLI / help / docstrings | “YOLO” generically; “YOLO26” when a concrete generation/model is meant |
| External export docs | Keep X-AnyLabel-toolkit path/script name as published upstream |

Files expected to change for naming (non-exhaustive): `README.md`, `CLAUDE.md`, `requirements.txt`, all script module docstrings / argparse help / print banners, template `.bat` REM headers, and `train_detector/train_detector.py` model strings + tables. New: `tile_yolo_dataset/`.

## Train auto-config + VRAM probe

### Defaults

- Replace every auto-selected `yolov8{m,l,x}.pt` with `yolo26{m,l,x}.pt`.
- Keep the same decision-table shape (image count × VRAM buckets → model + imgsz).
- Keep override flags: `--model`, `--imgsz`, `--batch`, `--workers`.

### Built-in fallbacks (FLOPs-scaled)

Published detect FLOPs @ 640 (Ultralytics YOLO26 vs YOLOv8):

| Size | YOLOv8 FLOPs (B) | YOLO26 FLOPs (B) | Ratio |
|---|---:|---:|---:|
| m | 78.9 | 68.2 | 0.864 |
| l | 165.2 | 86.4 | 0.523 |
| x | 257.8 | 193.9 | 0.752 |

Scale existing measured GB/image @ 1024 and round to two decimals:

| Model | Old (YOLOv8) | New fallback (YOLO26) |
|---|---:|---:|
| m | 0.60 | **0.52** |
| l | 1.10 | **0.58** |
| x | 1.60 | **1.20** |

Comment in code must state these are FLOPs-scaled from prior YOLOv8 measurements, not newly bench’d, and that the local probe cache overrides them when present.

Usable VRAM fraction remains **0.85**.

### Probe cache

- Path: `train_detector/weights/vram_estimates.json`
- Covered by existing gitignore rule `train_detector/weights/`
- Suggested schema:

```json
{
  "version": 1,
  "ultralytics": "x.y.z",
  "device_name": "NVIDIA ...",
  "updated_at": "ISO-8601",
  "estimates": {
    "yolo26m.pt": {"gb_per_image_1024": 0.51},
    "yolo26l.pt": {"gb_per_image_1024": 0.57},
    "yolo26x.pt": {"gb_per_image_1024": 1.18}
  }
}
```

- Load order for a model key: cache entry → else built-in fallback → else conservative default (current fallback for unknown models, historically ~1.10).

### When probe runs

1. **Fresh GPU train:** if CUDA is available and the cache lacks an entry for the chosen model, auto-probe that model before final batch selection, write/merge cache, then continue.
2. **`--probe-vram`:** force re-measure for `yolo26m/l/x` (or only `--model` if provided), write cache. Then: if `--input` is also provided, continue into normal training; otherwise exit 0 after writing (probe-only mode). `--input` remains required for any train path that is not probe-only.
3. **CPU / no CUDA:** skip probe; use built-in fallbacks.

### Probe method

- Resolve/download pretrained weights via existing `resolve_pretrained_weights()` into `train_detector/weights/`.
- No user dataset required for standalone `--probe-vram`.
- Short synthetic train step: fixed `imgsz=640`, small batch (2, fall back to 1 on OOM), tiny random tensor or minimal dataloader equivalent; one forward + backward (+ optimizer step if needed for realistic peak).
- Read peak allocated CUDA memory; convert to GB/image @ 1024 via `(1024/640)^2` scaling (same quadratic assumption as current batch math).
- Target runtime: seconds, not minutes.
- On probe failure: warn, keep/use built-ins, continue train when applicable.
- On corrupt/partial cache: ignore bad entries; re-probe missing/invalid models.

### Selection table

Do **not** rewrite tier thresholds in this pass. Only swap model names and VRAM numbers. Optional later follow-up: if probe cache shows x@1280 comfortably ≥ `MIN_BATCH` on 16GB, reconsider the x gate — out of scope now.

## Tiling (default on)

Recommended pipeline:

```text
raw / VOC / flat YOLO
        → dataset builders (whole images)
        → tile_yolo_dataset/   (default for large images / small objects)
        → train_detector/
        → detect_images/      (tiled by default; --no-tiles to opt out)
```

### Train prep — `tile_yolo_dataset/`

- **Input:** existing YOLO-format dataset (`data.yaml` + train/val/[test] images+labels).
- **Output:** new YOLO-format dataset of tiles (auto-versioned output dir like other scripts).
- **Defaults:**
  - `--imgsz` default **1024** (tile width/height)
  - overlap **20%** (`--overlap 0.2`)
  - if both image sides ≤ imgsz: copy through as a single tile (no slice)
- **Labels:** convert YOLO normalized → pixel; clip to tile; drop if remaining box area &lt; **20%** of original box area; re-normalize to tile size.
- **Empty tiles:** after labelled tiles are produced, keep empty tiles up to **~10% of total output tiles** (random sample among empties; if fewer empties exist, keep all). Write empty `.txt` label files.
- **Optional:** `tiles_manifest.json` under output (source path, tile xyxy offsets, split) for debug — lives under user `--output`, not the repo.
- **Template:** `_Run_tile_yolo_dataset_template.bat` (personal bats remain gitignored).
- Follow existing script patterns: `normalize_path()`, argparse style, auto-versioned non-empty outputs.

### Infer — `detect_images` tiled by default

- Default: tiled inference enabled.
- For each source image with width or height &gt; tile size: sliding window with same `imgsz` + 20% overlap → run detector per tile → map boxes to full-image pixel coords → class-wise NMS at IoU **0.5** → draw/export once on the full image.
- Images already ≤ tile size: single whole-image pass (no slice).
- **`--no-tiles`:** restore today’s whole-image path.
- **`--imgsz` / `--overlap`:** align with train-prep defaults when tiling (exact CLI names finalized in plan; must be documented).
- JSON sidecars and annotated images remain **full-frame** after merge (not per-tile outputs by default).
- Parallel batch/workers pipeline: tile crops are the inference units when tiling is on; merge happens per source image before save.

### Shared geometry

Prefer a small shared helper (either inside `tile_yolo_dataset/` imported carefully, or a tiny `yolo_tiling.py` used by both train-prep and detect) for:
- window grid generation (tile size + overlap)
- box clip + min-area filter
- box offset map tile → full image
- class-wise NMS

Avoid a heavy new package; keep it script-local and consistent with the repo’s flat script-folder style.

## Public-repo hygiene

### Keep (already correct)

- `*_personal.bat`
- `train_detector/runs/`, `train_detector/weights/`
- `detect_images/detections/`
- `exiftool/`
- `*.pt`, `*.onnx`, `*.engine`
- `*.yaml` (datasets; keep `!requirements*.txt` exception)
- `__pycache__/`, `.env`, remapped output name patterns
- `.claude/settings.local.json`

### Add

- `.cursor/` — local editor state (present locally, currently untracked and **not** ignored)
- Short comment that `vram_estimates.json` lives under ignored `train_detector/weights/`

### Commit policy for implementation

Stage only: public docs, source, `.gitignore`, `requirements.txt`, and this `docs/superpowers/` design/plan tree.  
Never stage: personal bats, weights, runs, exiftool binaries, probe cache, `.cursor`, datasets, tiled dataset outputs.

## Docs + requirements

### Docs

- `README.md` / `CLAUDE.md`: title, overview, script table, examples, auto-config tables, install blurb
- Document tiling in the main pipeline diagram and a dedicated section
- Training section: document auto-probe, `--probe-vram`, cache path
- Detect section: tiled-by-default + `--no-tiles`
- Template `.bat` REM lines: YOLOv8 → YOLO / YOLO26 as appropriate
- External X-AnyLabel pointer unchanged in path/script name
- Replace leftover `yolov8-toolkit` example paths with `yolov-toolkit` (GitHub name already updated)

### requirements.txt

```text
# Ultralytics YOLO (currently YOLO26; introduced in ultralytics 8.4.0)
# Keep updated with: pip install -U ultralytics
ultralytics>=8.4.0
```

Floor is **8.4.0** (Ultralytics YOLO26 Models Release). No upper pin — toolkit should float to latest via `pip install -U ultralytics`.

No new hard dependency required for tiling if implemented with Pillow/OpenCV + numpy already available via ultralytics/opencv-python. Prefer `opencv-python` already listed.

### Verification (light)

- No new full test suite required
- Check `--help` / argparse for `--probe-vram`, `--no-tiles`, tile script flags
- Code review of selection + probe load/merge/fallback paths
- Manual/logic check: clip+drop, empty-tile cap, NMS merge on a synthetic 2-tile overlap case
- Confirm `git check-ignore` covers `.cursor/`, `*_personal.bat`, `train_detector/weights/`, runs, exiftool

## Error handling summary

| Case | Behavior |
|---|---|
| No CUDA | Skip probe; use built-ins |
| Probe OOM at batch 2 | Retry batch 1; if still fail → warn + built-ins |
| Probe exception | Warn + built-ins; continue train |
| Corrupt JSON cache | Treat as missing; rebuild entries as needed |
| User passes `--model yolov8x.pt` | Allowed; probe/cache keyed by that filename if GPU path needs it |
| Image ≤ imgsz in tile prep | Copy through as one tile |
| Image ≤ imgsz in detect | Whole-image infer (no slice) |
| Tile with no labels after clip | Candidate for empty-tile pool |
| Corrupt/missing label file in tile prep | Skip pair with warning (match other dataset scripts’ harden style) |

## Non-goals / explicit non-changes

- Soft-NMS / WBF for tile merge (use class-wise NMS only)
- Auto-tiling inside `train_detector`
- Changing on-disk YOLO layout / `data.yaml` schema (tiled output still standard YOLO format)
- Baking tiling into VOC/VLM/flat-split/remap producers
- detect_images I/O parallelism redesign beyond fitting tiles into the existing batch/worker model
