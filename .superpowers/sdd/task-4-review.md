# Task 4 Review: Shared tile geometry (TDD)

Reviewed task package against base `30d6c4bbd09d6642ec30e55fe2105e4bcfd3829a` and the Task 4 brief. No source changes made.

## Verdicts

- Spec: ✅
- Quality: Approved

## Findings

No task-scoped issues found.

- All seven requested pure-helper interfaces are present.
- Tile windows validate the overlap, use the required stride calculation, and add flush-right/flush-bottom final windows.
- Box clipping returns tile-relative coordinates and excludes zero-area intersections; retained-area filtering uses the specified default 20% threshold.
- Empty-tile sampling is seeded and caps the default share at 10% of the combined labelled and empty output.
- NMS is greedy, confidence-ordered, per class, and defaults to IoU `0.5`.
- Tests cover the brief's required acceptance cases plus coordinate conversion round-tripping.

The implementation and task-specific test evidence in the report are sufficient; the full test suite was not re-run.
