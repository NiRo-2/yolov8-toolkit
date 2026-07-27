# Final Review — `feature/yolo-toolkit-latest-tiling`

**Reviewer:** Senior Code Reviewer (final pre-merge review)
**Date:** 2026-07-27
**Base (merge-base main):** `4532966a5e4508910307bc73146b361b10d48d7f`
**Head:** `2fe43684c6eaf4c80e9dc52b3661259c523ccc12`
**Range:** 8 commits, 19 files, +1048 / −135
**Design:** `docs/superpowers/specs/2026-07-27-yolo-toolkit-latest-design.md`
**Plan:** `docs/superpowers/plans/2026-07-27-yolo-toolkit-latest-tiling.md`

**Verdict: REQUEST CHANGES** — 3 small, well-scoped must-fix items. The branch is otherwise a faithful, high-quality implementation of the plan; none of the must-fix items require redesign.

---

## 1. Plan alignment

Every task in the plan is implemented, in the planned order, with the planned commit messages.

| Plan requirement | Status | Evidence |
|---|---|---|
| T1 `.gitignore` `.cursor/` + probe-cache comment | Done | `.gitignore:208`, `:222`; `git check-ignore -v` reports `.cursor`, `train_detector/weights/vram_estimates.json`, `*_personal.bat`, `weights/*.pt` all ignored |
| T1 `ultralytics>=8.4.0` + comments | Done | `requirements.txt` matches plan text verbatim |
| T2 `VRAM_PER_IMAGE` = m 0.52 / l 0.58 / x 1.20, FLOPs-scaled comment | Done | verified at runtime: `{'yolo26m.pt': 0.52, 'yolo26l.pt': 0.58, 'yolo26x.pt': 1.2}` |
| T2 auto-select returns `yolo26{m,l,x}.pt`; table shape unchanged | Done | `select_model_and_imgsz(2000,16) -> ('yolo26l.pt', 1024)`, `(500,16) -> ('yolo26m.pt',1280)` |
| T2 brand sweep, X-AnyLabeling path untouched | Done | `rg -i yolov8` outside `docs/` returns only the external `scripts/yolov8_pt_to_xanylabeling_onnx/` references and the intentional "FLOPs-scaled from prior YOLOv8 measurements" comment |
| T2 `yolov8-toolkit` → `yolov-toolkit` example path | Done | README export example |
| T3 `ESTIMATES_PATH` / load / save / get / probe / ensure / `--probe-vram` | Done | all six symbols present with the planned signatures |
| T3 CPU path skips cleanly, exits 0 without `--input` | Done | `--probe-vram --device cpu` → `[VRAM Probe] CUDA unavailable — using built-in FLOPs-scaled estimates`, exit 0 |
| T3 auto-probe on fresh GPU train before batch calc | Done | `auto_config()` calls `ensure_vram_estimates([model], …, force=False)` before `calc_batch` |
| T4 `tile_geometry.py` + `tests/test_tile_geometry.py` (TDD) | Done | `pytest tests/test_tile_geometry.py -q` → **6 passed** |
| T5 tile CLI with `--imgsz/--overlap/--empty-frac/--seed/--manifest`, auto-versioned output | Done | end-to-end smoke run below |
| T5 template `.bat` | Done | `tile_yolo_dataset/_Run_tile_yolo_dataset_template.bat` |
| T6 `--tiles/--no-tiles/--tile-imgsz/--tile-overlap/--tile-iou`, per-source batching, NMS merge | Done | `--help` shows all flags with stated defaults; stub-model run confirms offsets + merge |
| T7 README/CLAUDE docs, pipeline diagram, remote URL | Done | `git remote -v` → `https://github.com/NiRo-2/yolov-toolkit.git` |

Global constraints honored: brand "YOLO Toolkit"; "YOLO format" wording; m/l/x only; `ultralytics>=8.4.0` with no upper pin; tile defaults 1024 / 0.2 / 20% area / ~10% empties / NMS IoU 0.5 / `--no-tiles`. Non-goals respected: no soft-NMS/WBF, no auto-tiling inside `train_detector`, no tiling baked into the dataset producers, `data.yaml` schema unchanged.

**Verification I ran (read-only, no checkout mutation):**

- `pytest tests/test_tile_geometry.py -q` → 6 passed.
- `select_model_and_imgsz` / `calc_max_batch_for_imgsz` probed directly with no probe cache present.
- `python train_detector/train_detector.py --probe-vram --device cpu` → clean skip, exit 0.
- `python detect_images/detect_images.py --help` → all five tiling flags present.
- Synthetic tiling CLI run (2000×1024 train+val images, box straddling the tile seam, `--manifest`): 3 tiles per split, labels correctly clipped and re-normalized per tile, `data.yaml` and `tiles_manifest.json` written, small-image dataset copied through as a single tile.
- `detect_image_tiled` driven with a stub model: tile-local box `(10,20,40,60)` came back as `x1 ∈ {10, 829, 986}` (offsets applied), `imgsz=1024` passed on every call, single-window path used for a 500×500 image, non-contiguous crop views accepted by OpenCV.
- Peak-RSS measurement of the tiling CLI (see Important #1).

---

## 2. Critical findings

None. No data loss, no silent label corruption, no crash on the documented happy paths.

---

## 3. Important findings (must fix before merge: 3.1, 3.2, 3.3)

### 3.1 `tile_yolo_dataset` holds the entire split in RAM (must fix)

`collect_tiles()` builds every `TileRecord` for a whole split — including `image[y1:y2, x1:x2]` — and only `run()` writes them afterwards. Numpy slicing returns a **view**, so each record keeps its entire source image alive; peak memory therefore scales with the total decoded pixel volume of the split, not with one image.

Measured: 12 × 4000×3000 JPEGs (432 MB decoded) → **peak RSS 470 MB**. Extrapolated, a realistic 500-image 24 MP ortho set needs ~18 GB, and 1000 images ~36 GB — i.e. the script fails exactly on the large-image datasets it exists to serve.

Fix is local and small: collect only tile *metadata* (`split, source, window, labels`) in pass one, then write per source image (labelled tiles immediately; selected empties in a short second pass that re-reads that image), so at most one decoded image is resident.

### 3.2 Internal SDD scratch file committed to a public repo (must fix)

`4c34a4c` added `.superpowers/sdd/task-6-report.md` to the tree (`git ls-tree` confirms it at HEAD). The plan's commit policy allows only public docs, source, `.gitignore`, `requirements.txt` and the `docs/superpowers/` tree. The file also references a commit (`db690c8`) that does not exist on this branch, so it is stale as well as out of place.

Fix: `git rm --cached .superpowers/sdd/task-6-report.md` (ideally amended out of `4c34a4c`, or removed in a follow-up commit) and add `.superpowers/` to `.gitignore` so the rest of the scratch tree can't leak later.

### 3.3 Auto-config tables state an outcome the code no longer produces (must fix — triaged from Task 2 backlog)

Confirmed by running the shipped code with no probe cache: at 16 GB, `yolo26x@1280` yields batch **4** (`1.20 × 1.5625 = 1.875 GB/img`, `13.6 / 1.875 = 7 → 4`), which is below `MIN_BATCH = 8`, so `select_model_and_imgsz(6000, 16)` returns **`('yolo26x.pt', 1024)`**.

Two user-facing surfaces disagree with that:
- `train_detector.py` docstring table: `> 5,000 │ >= 16GB │ yolo26x │ 1280 │ x+1280 fits at batch 8 with 16GB` — wrong on both imgsz and batch. The `1,000–5,000` row's "l+1280 fits batch 4" is also stale (now 8).
- `README.md` auto-config table: `> 5,000 | ≥ 16GB | yolo26x | 1280`.

The inaccuracy predates this branch, but this branch is what changed the numbers, so it should not ship again. Fix the two table rows (and the reason text) to reflect the 1024 outcome, or note "1280 when batch ≥ 8 fits, else 1024".

### 3.4 Probe converts total peak memory into a per-image figure (should fix)

`probe_model_vram()` divides `torch.cuda.max_memory_allocated()` by `used_batch` (2, or 1 on OOM). Peak includes batch-independent cost — weights, gradients, optimizer states, plus Ultralytics' AMP check and the end-of-epoch validation pass — so the result is an *average*, not the *marginal* per-image cost, and at batch 2 the fixed overhead is roughly halved into it. For `yolo26x` (~1 GB of fixed fp32 weight/grad/optimizer state) the measured value can land well above the 1.20 built-in, and because the cache unconditionally overrides the built-ins, the "improvement" can silently make batch selection worse than shipping no probe at all.

This follows the design as written, so it is not a plan violation — but it is worth one cheap guard before users rely on it: measure at two batch sizes and take the slope, or clamp/warn when a measured value exceeds the built-in by more than ~2×. Cannot be validated in this environment (no CUDA).

### 3.5 Probe cache is never invalidated (should fix)

`save_vram_estimates()` records `ultralytics` version and `device_name`, but `load_vram_estimates()` reads only `estimates` and ignores both. Moving the checkout to a different GPU, or upgrading Ultralytics, silently reuses stale measurements. A version/device mismatch should drop the entries (or at least warn and suggest `--probe-vram`).

### 3.6 `CLAUDE.md` overstates tile batching (should fix)

`CLAUDE.md` says detect "batches all tiles from each source image together at `imgsz=tile`". The shipped code chunks a source image's tiles into groups of `batch_size` (`resolve_batch(args.batch)`, default 8 on CUDA and **1** on CPU) — verified with the stub model. Reword to "batches a source image's tiles in groups of `--batch`".

---

## 4. Minor findings / nits

1. **Flush-edge windows can near-duplicate their neighbour.** `iter_tile_windows` always appends `length - tile`. For 2000 px at tile 1024 / overlap 0.2 the starts are `0, 819, 976` — the last two windows overlap by 85%, producing two near-identical tiles with duplicated labels (and, in detect, a redundant inference). The design mandated the flush window, so this is conformant; consider dropping it when `final - positions[-1]` is a small fraction of the stride.
2. **Small images are re-encoded rather than copied through.** The design says an image ≤ imgsz is "copied through as a single tile"; `write_record` always goes through `cv2.imwrite`, which re-compresses JPEGs and drops EXIF. Harmless for training, but not what the design says.
3. **`data.yaml` `path:` key ignored.** Splits resolve relative to the dataset root only. Toolkit-generated datasets are fine; a Roboflow/Ultralytics yaml with `path: ../datasets/foo` would resolve to the wrong directory.
4. **Output `data.yaml` lists splits that produced no tiles.** `splits` is computed before collection and passed unchanged to `write_data_yaml`, so a declared-but-missing `test:` (warned and skipped) still appears in the tiled `data.yaml` pointing at a directory that does not exist.
5. **Tile-name collisions across extensions.** `a.jpg` and `a.png` in one split both map to label `a_x0_y0.txt`; the second write wins while both images survive, silently mislabelling one. Other scripts in this repo (`remap_yolo_labels`) already handle collisions explicitly.
6. **Model selection uses pre-probe estimates.** `select_model_and_imgsz()` runs before `ensure_vram_estimates()`, so the `x@1280` gate always uses built-ins while `calc_batch()` may use a measured value. Plan-conformant, but the two decisions can disagree; worth a comment.
7. **20% label-area drop rule is not in the docs prose** (triaged from Task 7 backlog — minor, fix with 3.3 while touching docs). Related wording bug in the same paragraph: README says empties are capped at "no more than 10% of the labelled-tile count", but the implementation caps them at 10% of *total output* (= 11.1% of labelled). One sentence fixes both.
8. **`pytest` is undeclared.** `tests/` ships but pytest is in neither `requirements.txt` nor any dev-requirements/doc note, and nothing in README/CLAUDE tells a contributor to run it.
9. **Default tiling changes small-image inference size.** With tiling on, single-window images now run at `imgsz=--tile-imgsz` (1024) instead of Ultralytics' 640 default — slower and not result-identical to `--no-tiles`. Documented in the module docstring, not in README.
10. **`get_vram_per_image()` re-reads the JSON on every call**, so a corrupt cache prints its warning repeatedly during one auto-config pass.
11. **Dead parameter:** `collect_tiles(..., dataset_root, ...)` is unused.
12. **README still claims "Python 3.8+"** while the code uses PEP 585 builtin generics in evaluated signatures (`dict[str, float]` in `train_detector.py`), requiring 3.9+. Pre-existing (`vlm_yolo_prep.py` already did this), not introduced here.

---

## 5. Testing gaps

- **No GPU validation of the probe.** `probe_model_vram()` has never been executed — no CUDA here. Unknowns that only a GPU run settles: whether Ultralytics accepts the all-empty-label synthetic dataset (it warns rather than fails, but that is version-dependent), whether the AMP check downloads an extra checkpoint mid-probe, and whether `torch.cuda.max_memory_allocated()` reads the intended device when `--device 1+` is used (it reports the *current* device). Recommend a one-time `--probe-vram` run on the target box, plus an eyeball of the resulting `vram_estimates.json` against the built-ins, before trusting auto-config.
- **No committed test for the tiled detect path** (Task 6 backlog item — confirmed real). The stub-model harnesses used during implementation and during this review were both throwaway. A `tests/test_detect_tiling.py` with a fake model would pin offset math, per-image call grouping, `imgsz=tile`, and the single-window shortcut in ~40 lines and needs no weights. Recommended, not blocking.
- **No real-weights end-to-end run of default-on tiling.** No `.pt` exists in the checkout. Since tiling is now the *default* path for every detect invocation, one real run on a wide image should be done before merge — specifically to look at seam behavior: class-wise NMS at IoU 0.5 will not merge an object split across two tiles into one box, which is the known accepted limitation of the design, but its practical severity is unmeasured.
- **No CLI-level test for `tile_yolo_dataset`.** Behavior is covered only by helper unit tests plus the ad-hoc runs in this review. The empty-tile cap, `--manifest`, auto-versioned output, and corrupt-input warnings have no committed coverage.
- **Corrupt/missing-input paths unexercised.** `[WARNING] Skipping corrupt image/label` branches were reviewed by reading, not run.

---

## 6. Merge recommendation

**Request changes** — small and well-scoped:

1. Stream tile writes so memory scales with one image, not one split (§3.1).
2. Remove `.superpowers/sdd/task-6-report.md` from the tree and ignore `.superpowers/` (§3.2).
3. Correct the `> 5,000 / ≥ 16GB` row (and the stale `l+1280` reason) in the `train_detector.py` docstring table and the README auto-config table (§3.3).

Then, in the same pass if convenient: §3.4–§3.6 and nits 7 and 8. Everything else can land as follow-ups.

Once those three are addressed, this is a merge. Plan alignment is essentially complete, the tiling geometry is correct and unit-tested, the naming sweep is clean (only the intentional external X-AnyLabeling path and the FLOPs-provenance comment retain "YOLOv8"), gitignore/remote hygiene checks pass, and the `--no-tiles` legacy path is preserved intact.
