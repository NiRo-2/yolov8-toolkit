# Final Fix Pass — must-fix findings from whole-branch review

Branch: `feature/yolo-toolkit-latest-tiling`
Ref reviewed: `docs review` in `.superpowers/sdd/final-review.md` (§3.1–3.3, plus §3.6 and nit #7)

## 1. `tile_yolo_dataset` memory (must fix §3.1)

Rewrote `tile_yolo_dataset/tile_yolo_dataset.py`:
- Replaced `collect_tiles()` / `TileRecord` (held every tile's numpy crop for the
  whole split) with `process_split()` / `TileMeta` (coordinates only, no pixel data).
- Pass 1: decode one source image at a time, write labelled tiles to disk
  immediately, drop the decoded array at end of each image's loop iteration.
  Empty-tile candidates keep only `TileMeta` (split/source/coords/labels=[]).
- Pass 2: after the whole split is scanned (so `labelled_count` is known),
  `select_empty_tiles()` picks the capped empty indices; those are grouped by
  source image and each source is re-decoded once to extract + write just the
  selected crops.
- Net effect: peak resident memory is now bounded by one decoded source image
  (+ its tiles) at a time, not the whole split's decoded pixel volume.
- `run()` updated to call `process_split()` per split; `write_record` renamed
  to `write_tile()` (function, not tied to a record dataclass).

## 2. Removed committed scratch (must fix §3.2)

- `git rm --cached .superpowers/sdd/task-6-report.md`
- Added `.superpowers/sdd/` to `.gitignore` (with a comment noting
  `docs/superpowers/` — specs/plans — stays tracked; only the scratch working
  dir is ignored).
- `git ls-files .superpowers/sdd` → empty (verified).

## 3. Auto-config table accuracy (must fix §3.3)

- `train_detector/train_detector.py` docstring decision table:
  - `1,000-5,000 / >=16GB` row: corrected stale "l+1280 fits batch 4" → "fits
    batch 8" (current `VRAM_PER_IMAGE["yolo26l.pt"]=0.58` math gives batch 8
    at 16GB, not 4).
  - `> 5,000 / >=16GB` row: `yolo26x@1280` marked with `†`; added a footnote
    explaining it only ships when `calc_max_batch_for_imgsz(...) >= MIN_BATCH
    (8)`, which needs ~17.6GB+; at exactly 16GB it yields batch 4 and falls
    back to 1024 (verified: `1.20 * 1.5625 = 1.875 GB/img`, `13.6 / 1.875 = 7
    -> bucketed to 4 < MIN_BATCH`).
- `README.md` auto-config table: same two rows corrected, same `†` footnote.

## 4. Should-fix items (included)

- `CLAUDE.md` (§3.6): corrected "batches all tiles from each source image
  together at imgsz=tile" → "batches each source image's tiles in chunks of
  `--batch` (default 8 CUDA / 1 CPU) ... tiles from different source images
  are never batched together" — matches `detect_image_tiled()`'s chunked
  `model(chunk, ...)` calls.
- `README.md` / `CLAUDE.md` (nit #7): documented the 20%-remaining-area rule
  for clipped boxes (`keep_clipped_box(..., min_frac=0.2)`), and corrected the
  empty-tile description from "no more than 10% of the labelled-tile count"
  to "no more than 10% of the **total output tile count**" (~11% of labelled
  tiles) in both files.
- `CLAUDE.md` tile_yolo_dataset one-liner also now mentions the per-image
  streaming behavior from fix #1.

## Verification

- `pytest tests/test_tile_geometry.py -q` → **6 passed**.
- Smoke test: synthetic 4-image dataset (train: 3 images, 1 with box, 1 empty
  label file, 1 missing label file; val: 1 image, no label file) built with
  OpenCV/numpy, then tiled:
  - `--empty-frac 0.10`: `[train] 7 labelled + 0 empty tiles written`,
    `[val] 0 labelled + 0 empty tiles written` (val has no labelled tiles, so
    `select_empty_tiles` correctly yields 0 empties there); output images,
    label files, `data.yaml`, and `tiles_manifest.json` all written correctly;
    spot-checked a clipped box's re-normalized coordinates by hand — correct.
  - `--empty-frac 0.50` (forces the empty-tile second pass to actually run):
    `[train] 7 labelled + 5 empty tiles written`; verified an empty-tile crop
    on disk decodes to the expected `(1024, 1024, 3)` shape and its label file
    is empty — confirms the re-decode-by-source second pass works.
  - Temp dataset/output dirs removed after verification.
- `git ls-files .superpowers/sdd` → empty.

## Concerns / follow-ups not in scope

- Should-fix §3.4 (probe VRAM measures average not marginal per-image cost)
  and §3.5 (probe cache never invalidated on GPU/Ultralytics version change)
  were not addressed — out of the "quick" should-fix bar, left as follow-ups.
- Minor nits #1–6, #8–12 from the review (flush-edge near-duplicate tiles,
  small images re-encoded instead of copied, `data.yaml` `path:` key ignored,
  declared-but-empty splits still listed, cross-extension tile-name
  collisions, no committed tiled-detect test, `pytest` undeclared as a dev
  dependency, etc.) were left as-is per the "must fix" + "quick should-fix"
  scope of this pass.
