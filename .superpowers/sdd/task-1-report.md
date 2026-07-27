# Task 1 Report: Public hygiene + requirements floor

**Status:** DONE_WITH_CONCERNS  
**Branch:** `feature/yolo-toolkit-latest-tiling`  
**Date:** 2026-07-27

## Summary

Updated `.gitignore` to ignore `.cursor/` and documented `train_detector/weights/` probe cache; raised `requirements.txt` floor to `ultralytics>=8.4.0` with YOLO26-oriented comments.

## Changes

### `.gitignore`

- Added after the existing Cursor block (after `.cursorindexingignore`):
  - Comment: `# Cursor local project state (do not commit)`
  - Pattern: `.cursor/`
- Script-local outputs block (`train_detector/runs/`, `train_detector/weights/`, `detect_images/detections/`) was already present; added a standalone comment line `# includes vram_estimates.json probe cache` immediately above `train_detector/weights/` (see Concerns).

### `requirements.txt`

Replaced contents per plan:

- Header comments referencing YOLO26 and `ultralytics 8.4.0`
- `ultralytics>=8.4.0` (was unpinned `ultralytics`)
- Unchanged: `opencv-python`, `psutil`, `requests`, `pillow`, `pyyaml`

## Verification

```text
$ git check-ignore -v .cursor train_detector/weights/vram_estimates.json
.gitignore:208:.cursor/	.cursor
.gitignore:222:train_detector/weights/	train_detector/weights/vram_estimates.json
```

Exit code: 0. Both paths reported as ignored.

Environment: `git version 2.53.0.windows.1`

## Commit

| SHA (short) | Subject |
|-------------|---------|
| `a0d02fd` | Ignore .cursor and require ultralytics>=8.4.0 for YOLO26. |

Files committed: `.gitignore`, `requirements.txt` only.

Not committed (intentionally): `.superpowers/`, personal bats, weights, `.cursor/`.

## Self-review

- Task scope limited to `.gitignore` and `requirements.txt` — no script or README edits.
- `.cursor/` placement matches brief (after Cursor comment block).
- Requirements text matches brief verbatim.
- `git check-ignore` passes for both required paths.
- No push performed.

## Concerns

**Inline gitignore comment:** The plan shows `train_detector/weights/   # includes vram_estimates.json probe cache` on one line. On Git 2.53 (Windows), spaces before `#` become part of the ignore pattern, so `train_detector/weights/vram_estimates.json` is **not** ignored with that formatting. Verified by toggling the inline form (check-ignore exit 1) vs plain `train_detector/weights/` (exit 0).

**Mitigation:** Used the same comment text on the line immediately above `train_detector/weights/` so matching behavior matches the brief’s intent and verification passes.

If the project standardizes on inline gitignore comments elsewhere, consider documenting “no spaces before `#` on pattern lines” or using end-of-line comments only when the pattern has no trailing slash.

## Next task readiness

Task 2+ can assume `ultralytics>=8.4.0` in requirements and that `.cursor/` plus script-local outputs remain gitignored.
