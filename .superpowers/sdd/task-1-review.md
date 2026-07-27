# Task 1 Review: Public hygiene + requirements floor

**Reviewer:** task-scoped gate  
**Base:** `4532966a5e4508910307bc73146b361b10d48d7f`  
**Head:** `a0d02fd052030a8ec17d47eed5329945d65cac42`  
**Date:** 2026-07-27

## Verdicts

| Gate | Result |
|------|--------|
| **Spec compliance** | ✅ |
| **Task quality** | **Approved** |

## Spec compliance

All task steps are satisfied against `task-1-brief.md` and global constraints:

- **`.gitignore`:** `.cursor/` added with the specified header comment immediately after the existing Cursor block (after `.cursorindexingignore`). Script-local outputs were already present; a single `train_detector/weights/` entry remains, with probe-cache documentation added.
- **`requirements.txt`:** Matches the brief verbatim (`ultralytics>=8.4.0`, no upper pin; YOLO26-oriented header comments; unchanged secondary deps).
- **Verification:** Reported `git check-ignore -v` output correctly shows `.cursor` and `train_detector/weights/vram_estimates.json` ignored via `.cursor/` and `train_detector/weights/` rules.
- **Commit:** Message and file set (`.gitignore`, `requirements.txt` only) match Step 4; no personal bats, weights, runs, exiftool, `.cursor`, or datasets in the commit.
- **Scope:** No out-of-scope brand/tiling/script changes; YOLO26 mentions are limited to comments/commit text prescribed by the brief.

**Note (not a compliance gap):** The brief’s example puts `# includes vram_estimates.json probe cache` on the same line as `train_detector/weights/`. The implementation uses a standalone comment line above the pattern so Git does not treat spaces before `#` as part of the path. Behavior and acceptance checks align with the brief’s intent.

## Task quality

### Critical

None.

### Important

None.

### Minor

1. **Gitignore comment style:** Probe-cache note is on the line above `train_detector/weights/` rather than inline. This is the correct fix for Git path rules; future editors should not “simplify” to the brief’s one-line example without re-running `git check-ignore`.
2. **Transparency:** `DONE_WITH_CONCERNS` and the inline-comment pitfall in the implementer report are appropriate and useful for downstream tasks.
3. **Review artifact:** `task-1-review-pkg.md` contained an empty diff body; head commit was confirmed via `git diff`/`git show` (read-only).

## Summary

Task 1 is a minimal, correct hygiene commit: public repo ignores `.cursor/` and documents the weights probe cache under existing script-local rules, and `requirements.txt` enforces `ultralytics>=8.4.0` as specified. The only deliberate deviation from the brief’s literal gitignore snippet is well justified and keeps `vram_estimates.json` under `train_detector/weights/` coverage. **Approved** for task gate; safe baseline for Task 2+.
