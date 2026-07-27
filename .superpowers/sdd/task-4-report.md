# Task 4 Report: Shared tile geometry

## Status

DONE

## Scope completed

- Added the pure shared geometry module at
  `tile_yolo_dataset/tile_geometry.py`.
- Added a package marker so consumers can import it as
  `tile_yolo_dataset.tile_geometry`.
- Implemented tile-window generation, YOLO/pixel coordinate conversion, tile
  clipping and retained-area filtering, deterministic empty-tile selection,
  and per-class greedy xyxy NMS.
- Added six unit tests covering the required acceptance cases plus coordinate
  conversion round-tripping.
- Did not build a CLI or change inference/training integration.

## TDD evidence

### RED

Before the package/module existed, `pytest tests/test_tile_geometry.py -v`
failed during collection with the expected
`ModuleNotFoundError: No module named 'tile_yolo_dataset'`.

### GREEN

After implementation:

- `PYTHONPATH=<repo root> pytest tests/test_tile_geometry.py -v`: **6 passed**
  in 0.01s.
- `python -m pytest -v`: **6 passed** in 0.09s.
- `python -m py_compile tile_yolo_dataset/tile_geometry.py`: exited 0.
- `git diff --check`: exited 0.
- IDE diagnostics for the new module and tests: no errors.

## Commit

`47f5d3e Add shared tile geometry helpers with unit tests.`

## Concerns

- In this PowerShell environment, bare `pytest` does not add the repository
  root to `sys.path`; invoking it with `PYTHONPATH` set to the repository root
  (or using `python -m pytest`) resolves imports. The package itself is
  importable normally.
