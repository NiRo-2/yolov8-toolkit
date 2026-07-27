# YOLO Toolkit Latest + Tiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate this public toolkit to version-agnostic **YOLO Toolkit** defaults on **YOLO26**, add local VRAM auto-probe, default-on train/infer tiling for small objects on large images, and harden gitignore/remote hygiene.

**Architecture:** Keep the existing per-script folder layout. Shared tiling math lives in `tile_yolo_dataset/tile_geometry.py` and is imported by both the train-prep CLI and `detect_images` (same `sys.path` pattern as `ortho_tag_sidecar`). Training keeps auto-config in `train_detector.py` with FLOPs-scaled fallbacks overridden by a gitignored JSON probe cache under `train_detector/weights/`.

**Tech Stack:** Python 3, ultralytics>=8.4.0, OpenCV, Pillow, PyYAML, pytest (new, for pure tiling helpers only).

## Global Constraints

- Brand: **YOLO Toolkit**; current generation note: Ultralytics YOLO (currently **YOLO26**)
- Dataset wording: **YOLO format** (never “YOLOv8 format”)
- Auto models: `yolo26m.pt` / `yolo26l.pt` / `yolo26x.pt` only
- `ultralytics>=8.4.0` (no upper pin)
- Probe cache: `train_detector/weights/vram_estimates.json` (gitignored via `weights/`)
- Tiling defaults: tile=`imgsz` (1024), overlap=0.2, clip+drop if area&lt;20%, empty tiles capped ~10%, infer NMS IoU=0.5, `--no-tiles` opt-out
- Public commits only: no personal bats, weights, runs, exiftool, `.cursor`, datasets
- Spec: `docs/superpowers/specs/2026-07-27-yolo-toolkit-latest-design.md`

**Note:** Spec has two product slices (YOLO26 migration + tiling). This is one sequenced plan: Tasks 1–3 ship migration without tiling; Tasks 4–6 add tiling; Task 7 finishes docs/remote.

## File map

| File | Responsibility |
|---|---|
| `.gitignore` | Ignore `.cursor/`; comment probe cache coverage |
| `requirements.txt` | `ultralytics>=8.4.0` + comments |
| `README.md`, `CLAUDE.md` | Brand, YOLO26, tiling pipeline, probe docs |
| `*_template.bat` + script headers | YOLOv8 → YOLO / YOLO26 wording |
| `train_detector/train_detector.py` | YOLO26 defaults, VRAM table, probe, `--probe-vram` |
| `tile_yolo_dataset/tile_geometry.py` | Window grid, clip labels, NMS, empty-cap helpers |
| `tile_yolo_dataset/tile_yolo_dataset.py` | Train-prep CLI |
| `tile_yolo_dataset/_Run_tile_yolo_dataset_template.bat` | Template runner |
| `detect_images/detect_images.py` | Default tiled infer + merge; `--no-tiles` |
| `tests/test_tile_geometry.py` | Pure-function tests for tiling math |

---

### Task 1: Public hygiene + requirements floor

**Files:**
- Modify: `.gitignore`
- Modify: `requirements.txt`
- Test: shell `git check-ignore`

**Interfaces:**
- Consumes: none
- Produces: ignored `.cursor/`; `ultralytics>=8.4.0` install floor

- [ ] **Step 1: Update `.gitignore`**

Add after the existing Cursor comment block (near `.cursorignore`):

```gitignore
# Cursor local project state (do not commit)
.cursor/

# Script-local generated outputs (see README "Local outputs")
train_detector/runs/
train_detector/weights/   # includes vram_estimates.json probe cache
detect_images/detections/
```

(If `train_detector/weights/` is already listed once, keep a single entry and only add the inline comment + `.cursor/`.)

- [ ] **Step 2: Update `requirements.txt`**

Replace file contents with:

```text
# Ultralytics YOLO (currently YOLO26; introduced in ultralytics 8.4.0)
# Keep updated with: pip install -U ultralytics
ultralytics>=8.4.0
opencv-python
psutil

# VLM dataset preparation (vlm_yolo_prep.py)
requests
pillow
pyyaml
```

- [ ] **Step 3: Verify ignore**

Run:

```bash
git check-ignore -v .cursor train_detector/weights/vram_estimates.json
```

Expected: both paths reported as ignored (weights via `train_detector/weights/` rule).

- [ ] **Step 4: Commit**

```bash
git add .gitignore requirements.txt
git commit -m "Ignore .cursor and require ultralytics>=8.4.0 for YOLO26."
```

---

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

### Task 3: VRAM probe + cache in train_detector

**Files:**
- Modify: `train_detector/train_detector.py`
- Test: manual `--probe-vram` on CUDA if available; CPU path must skip cleanly

**Interfaces:**
- Consumes: `VRAM_PER_IMAGE` built-ins from Task 2; `resolve_pretrained_weights(model_name: str) -> Path`
- Produces:
  - `ESTIMATES_PATH: Path` = `WEIGHTS_DIR / "vram_estimates.json"`
  - `load_vram_estimates() -> dict[str, float]`  # model -> gb_per_image_1024
  - `save_vram_estimates(estimates: dict[str, float], device_name: str) -> None`
  - `get_vram_per_image(model: str) -> float`  # cache then built-in then 1.10
  - `probe_model_vram(model_name: str, device: str = "0") -> float`
  - `ensure_vram_estimates(models: list[str], device: str, force: bool = False) -> dict[str, float]`
  - CLI `--probe-vram`

- [ ] **Step 1: Add estimate load/save/get helpers**

Place near `VRAM_PER_IMAGE`:

```python
ESTIMATES_PATH = WEIGHTS_DIR / "vram_estimates.json"


def load_vram_estimates() -> dict[str, float]:
    if not ESTIMATES_PATH.exists():
        return {}
    try:
        with open(ESTIMATES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        out: dict[str, float] = {}
        for name, meta in (data.get("estimates") or {}).items():
            val = meta.get("gb_per_image_1024") if isinstance(meta, dict) else None
            if isinstance(val, (int, float)) and val > 0:
                out[str(name)] = float(val)
        return out
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
        print(f"[WARNING] Ignoring corrupt VRAM cache {ESTIMATES_PATH}: {e}")
        return {}


def save_vram_estimates(estimates: dict[str, float], device_name: str) -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    merged = load_vram_estimates()
    merged.update(estimates)
    try:
        import ultralytics
        ultra_ver = getattr(ultralytics, "__version__", "unknown")
    except Exception:
        ultra_ver = "unknown"
    from datetime import datetime, timezone
    payload = {
        "version": 1,
        "ultralytics": ultra_ver,
        "device_name": device_name,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "estimates": {
            k: {"gb_per_image_1024": round(v, 4)} for k, v in sorted(merged.items())
        },
    }
    with open(ESTIMATES_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def get_vram_per_image(model: str) -> float:
    cached = load_vram_estimates().get(model)
    if cached is not None:
        return cached
    return float(VRAM_PER_IMAGE.get(model, 1.10))
```

Add `import json` at top if missing. Replace `VRAM_PER_IMAGE.get(model, 1.10)` usages inside `calc_max_batch_for_imgsz` and `calc_batch` with `get_vram_per_image(model)`.

- [ ] **Step 2: Implement `probe_model_vram`**

```python
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch


def _write_probe_dataset(root: Path, imgsz: int = 640) -> Path:
    img_dir = root / "images"
    lbl_dir = root / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "probe.jpg"), img)
    (lbl_dir / "probe.txt").write_text("", encoding="utf-8")
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        "train: images\nval: images\nnc: 1\nnames: ['obj']\n",
        encoding="utf-8",
    )
    return yaml_path


def probe_model_vram(model_name: str, device: str = "0") -> float:
    """Return measured GB/image @1024 from a tiny synthetic train step @640."""
    weights = resolve_pretrained_weights(model_name)
    model = YOLO(str(weights))
    used_batch = 1
    with tempfile.TemporaryDirectory(prefix="yolo_vram_probe_") as tmp:
        tmp_path = Path(tmp)
        data_yaml = _write_probe_dataset(tmp_path, imgsz=640)
        last_err: Exception | None = None
        for batch in (2, 1):
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                model.train(
                    data=str(data_yaml),
                    epochs=1,
                    imgsz=640,
                    batch=batch,
                    device=device,
                    project=str(tmp_path / "runs"),
                    name="probe",
                    exist_ok=True,
                    verbose=False,
                    plots=False,
                    save=False,
                    workers=0,
                    patience=0,
                )
                used_batch = batch
                last_err = None
                break
            except torch.cuda.OutOfMemoryError as e:
                last_err = e
                torch.cuda.empty_cache()
        if last_err is not None:
            raise last_err
        peak_bytes = float(torch.cuda.max_memory_allocated())
    gb_per_img_640 = (peak_bytes / (1024 ** 3)) / float(used_batch)
    return gb_per_img_640 * ((1024 / 640) ** 2)
```

- [ ] **Step 3: Wire `ensure_vram_estimates` + CLI**

```python
def cuda_available_for(device: str) -> bool:
    if str(device).lower() == "cpu":
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def ensure_vram_estimates(models: list[str], device: str, force: bool = False) -> dict[str, float]:
    if not cuda_available_for(device):
        print("[VRAM Probe] CUDA unavailable — using built-in FLOPs-scaled estimates")
        return {m: get_vram_per_image(m) for m in models}
    cached = load_vram_estimates()
    need = [m for m in models if force or m not in cached]
    if not need:
        return {m: get_vram_per_image(m) for m in models}
    print(f"[VRAM Probe] Measuring: {', '.join(need)}")
    measured: dict[str, float] = {}
    try:
        import torch
        device_name = torch.cuda.get_device_name(0)
    except Exception:
        device_name = "cuda"
    for m in need:
        try:
            measured[m] = probe_model_vram(m, device=device)
            print(f"  {m}: {measured[m]:.3f} GB/img @1024")
        except Exception as e:
            print(f"[WARNING] Probe failed for {m}: {e} — using built-in fallback")
    if measured:
        save_vram_estimates(measured, device_name)
    return {m: get_vram_per_image(m) for m in models}
```

Argparse:

```python
parser.add_argument(
    "--probe-vram", action="store_true",
    help="Measure/refresh local VRAM estimates (yolo26m/l/x or --model); exit if --input omitted",
)
```

In `main` / entry:

- If `--probe-vram`: models = `[args.model]` if args.model else `["yolo26m.pt","yolo26l.pt","yolo26x.pt"]`; call `ensure_vram_estimates(..., force=True)`; if not `args.input` and not `args.resume`: `sys.exit(0)`.
- In `train()` after `auto_config` model is known (or inside `auto_config` before batch calc): if CUDA, `ensure_vram_estimates([model], device=args.device, force=False)` so missing cache entries are filled before `calc_batch`.

Refactor `auto_config` so batch math uses `get_vram_per_image` **after** optional ensure for the selected model.

- [ ] **Step 4: Verify help + CPU skip**

Run:

```bash
python train_detector/train_detector.py --help
python train_detector/train_detector.py --probe-vram --device cpu
```

Expected: help shows `--probe-vram`; CPU probe prints skip message and exits 0 without requiring `--input`.

- [ ] **Step 5: Commit**

```bash
git add train_detector/train_detector.py
git commit -m "Add local VRAM probe cache for YOLO26 batch auto-config."
```

---

### Task 4: Shared tile geometry (TDD)

**Files:**
- Create: `tile_yolo_dataset/tile_geometry.py`
- Create: `tests/test_tile_geometry.py`
- Create: `tile_yolo_dataset/__init__.py` (empty) only if needed for imports; prefer path insert like ortho_tag_sidecar and keep module importable as `tile_geometry`

**Interfaces:**
- Consumes: none
- Produces:
  - `iter_tile_windows(width: int, height: int, tile: int, overlap: float) -> list[tuple[int,int,int,int]]`  # x1,y1,x2,y2
  - `yolo_line_to_xyxy(parts: list[float], img_w: int, img_h: int) -> tuple[int, float,float,float,float]`  # cls,x1,y1,x2,y2
  - `xyxy_to_yolo_line(cls_id: int, x1: float, y1: float, x2: float, y2: float, tile_w: int, tile_h: int) -> str`
  - `clip_box_to_tile(x1,y1,x2,y2, tx1,ty1,tx2,ty2) -> tuple[float,float,float,float] | None`
  - `keep_clipped_box(orig_area: float, clipped: tuple[float,float,float,float], min_frac: float = 0.2) -> bool`
  - `select_empty_tiles(labelled_count: int, empty_indices: list[int], empty_frac: float = 0.10, seed: int = 42) -> list[int]`
  - `nms_xyxy(dets: list[dict], iou_thresh: float = 0.5) -> list[dict]`
    - each det: `{"cls": int, "conf": float, "x1": float, "y1": float, "x2": float, "y2": float}`

- [ ] **Step 1: Write failing tests**

Create `tests/test_tile_geometry.py`:

```python
from tile_yolo_dataset.tile_geometry import (
    iter_tile_windows,
    keep_clipped_box,
    clip_box_to_tile,
    select_empty_tiles,
    nms_xyxy,
)


def test_iter_tile_windows_single_when_small():
    assert iter_tile_windows(800, 600, tile=1024, overlap=0.2) == [(0, 0, 800, 600)]


def test_iter_tile_windows_overlap_grid():
    wins = iter_tile_windows(2000, 1024, tile=1024, overlap=0.2)
    assert wins[0] == (0, 0, 1024, 1024)
    assert wins[-1][2] == 2000
    assert len(wins) >= 2


def test_keep_clipped_box_drops_small_remainder():
    clipped = clip_box_to_tile(0, 0, 100, 100, 90, 0, 190, 100)
    assert clipped is not None
    assert keep_clipped_box(100 * 100, clipped, min_frac=0.2) is False


def test_select_empty_tiles_caps_fraction():
    chosen = select_empty_tiles(labelled_count=90, empty_indices=list(range(50)), empty_frac=0.10, seed=0)
    # total output would be 90 + len(chosen); empty share ~= 10% of total
    total = 90 + len(chosen)
    assert len(chosen) / total <= 0.10 + 1e-9
    assert len(chosen) >= 1


def test_nms_xyxy_keeps_higher_conf_same_class():
    dets = [
        {"cls": 0, "conf": 0.9, "x1": 0, "y1": 0, "x2": 10, "y2": 10},
        {"cls": 0, "conf": 0.5, "x1": 1, "y1": 1, "x2": 11, "y2": 11},
        {"cls": 1, "conf": 0.8, "x1": 0, "y1": 0, "x2": 10, "y2": 10},
    ]
    out = nms_xyxy(dets, iou_thresh=0.5)
    assert len(out) == 2
    assert {(d["cls"], round(d["conf"], 1)) for d in out} == {(0, 0.9), (1, 0.8)}
```

- [ ] **Step 2: Run tests — expect fail**

```bash
pip install pytest -q
cd /d/Nir/DevProjects/yolov-toolkit
pytest tests/test_tile_geometry.py -v
```

Expected: import or attribute failures.

- [ ] **Step 3: Implement `tile_geometry.py`**

Implement all functions listed in Interfaces. Rules:

- `overlap` in `[0, 1)`; stride = `max(1, int(tile * (1 - overlap)))`.
- Always include a final window flush-right / flush-bottom so `x2==width` / `y2==height` when image larger than tile.
- If `width <= tile and height <= tile`: exactly one window `(0,0,width,height)`.
- `clip_box_to_tile`: intersect boxes; return coords relative to tile origin `(x-tx1, y-ty1)`; `None` if no intersection.
- `select_empty_tiles`: max empties `floor(labelled_count * empty_frac / (1 - empty_frac))` when `empty_frac < 1`, else 0 if labelled_count==0 keep none (or keep up to a small cap of 0 — labelled required). Shuffle with `random.Random(seed)`.
- `nms_xyxy`: per-class greedy NMS by confidence.

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_tile_geometry.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tile_yolo_dataset/tile_geometry.py tests/test_tile_geometry.py
git commit -m "Add shared tile geometry helpers with unit tests."
```

---

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

### Task 6: Default tiled inference in `detect_images`

**Files:**
- Modify: `detect_images/detect_images.py`
- Test: logic smoke with `--no-tiles` unchanged path; tiled path on a wide synthetic image if a tiny model is available (optional)

**Interfaces:**
- Consumes: `iter_tile_windows`, `nms_xyxy` from `tile_geometry`
- Produces: CLI flags `--tiles/--no-tiles` (default tiles on), `--tile-imgsz` (default 1024), `--tile-overlap` (default 0.2), `--tile-iou` (default 0.5)

- [ ] **Step 1: Import tile_geometry**

Near ortho_tag_sidecar import:

```python
_TILE_DIR = _REPO_ROOT / "tile_yolo_dataset"
if str(_TILE_DIR) not in sys.path:
    sys.path.insert(0, str(_TILE_DIR))
from tile_geometry import iter_tile_windows, nms_xyxy  # type: ignore
```

- [ ] **Step 2: Add argparse flags**

```python
parser.add_argument("--tiles", dest="tiles", action="store_true", help="Tile large images before detect (default: True)")
parser.add_argument("--no-tiles", dest="tiles", action="store_false", help="Disable tiling; whole-image inference")
parser.set_defaults(tiles=True)
parser.add_argument("--tile-imgsz", type=int, default=1024, help="Tile size in pixels when --tiles is on (default: 1024)")
parser.add_argument("--tile-overlap", type=float, default=0.2, help="Tile overlap fraction (default: 0.2)")
parser.add_argument("--tile-iou", type=float, default=0.5, help="NMS IoU for merging tiled detections (default: 0.5)")
```

Update module docstring accordingly.

- [ ] **Step 3: Add `detect_image_tiled(model, image_bgr, conf, tile, overlap, iou) -> results-like structure`**

Implementation sketch:

1. `h, w = image.shape[:2]`
2. If not tiling or both sides ≤ tile: `return list(model([image], conf=conf, verbose=False))[0]` (existing behavior).
3. Else: for each window, crop, `model(crop)`, map each box `x1+=tx1` etc., collect dicts `{cls, conf, x1,y1,x2,y2}`.
4. `kept = nms_xyxy(all_dets, iou_thresh=iou)`.
5. Build a lightweight namespace/object compatible with existing `draw_detections` / `export_json` **or** refactor those helpers to accept a simple list of dets + class names.

Prefer minimal invasive approach: create a small `SimpleNamespace`/`Boxes` shim only if needed; otherwise refactor `draw_detections` and `export_json` to accept:

```python
# dets: list[{"cls": int, "conf": float, "x1": float, "y1": float, "x2": float, "y2": float}]
```

and keep a thin adapter from ultralytics Results → that list for the non-tiled path.

- [ ] **Step 4: Integrate into batch loop**

When `args.tiles` is True, per-image tiled detect (tiles may have different counts — do **not** batch unrelated source images’ tiles across different sources in v1). Still use ThreadPoolExecutor for post_process. When `args.tiles` is False, keep today’s `model(images_for_model, ...)` batching.

- [ ] **Step 5: Verify help**

```bash
python detect_images/detect_images.py --help
```

Expected: shows `--no-tiles`, `--tile-imgsz`, defaults documented.

- [ ] **Step 6: Commit**

```bash
git add detect_images/detect_images.py
git commit -m "Enable default tiled detection with full-image NMS merge."
```

---

### Task 7: Docs, CLAUDE, remote URL polish

**Files:**
- Modify: `README.md`, `CLAUDE.md`
- Shell: `git remote set-url` if still on old URL

**Interfaces:**
- Consumes: behaviors from Tasks 1–6
- Produces: accurate public docs + correct `origin`

- [ ] **Step 1: Update README pipeline**

Include:

```text
raw/VOC/flat → dataset builders → tile_yolo_dataset (recommended) → train_detector → detect_images (tiled by default)
```

Document `--probe-vram`, cache path, tiling flags, `ultralytics>=8.4.0`, brand YOLO Toolkit / currently YOLO26. Fix any `yolov8-toolkit` path examples to `yolov-toolkit`.

- [ ] **Step 2: Update CLAUDE.md** scripts table + key details to match.

- [ ] **Step 3: Fix git remote if needed**

```bash
git remote -v
git remote set-url origin https://github.com/NiRo-2/yolov-toolkit.git
git remote -v
```

Expected: fetch/push URLs end with `yolov-toolkit.git`.

- [ ] **Step 4: Final ignore audit**

```bash
git status --ignored -s | head
git check-ignore -v .cursor detect_images/_Run_detect_images_personal.bat train_detector/weights/yolo26n.pt
```

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "Document YOLO26 defaults, VRAM probe, and tiling pipeline."
```

---

## Self-review (plan vs spec)

| Spec requirement | Task |
|---|---|
| Brand YOLO Toolkit / YOLO format / YOLO26 note | 2, 7 |
| `yolo26m/l/x` defaults + FLOPs VRAM fallbacks | 2 |
| Auto probe + `--probe-vram` + cache JSON | 3 |
| `.cursor/` gitignore + public commit hygiene | 1, 7 |
| `ultralytics>=8.4.0` | 1 |
| Remote URL + `yolov8-toolkit` path fix | 7 |
| `tile_yolo_dataset` train prep | 4, 5 |
| detect tiled default + NMS + `--no-tiles` | 4, 6 |
| Empty tile ~10% cap / clip 20% / overlap 20% | 4, 5 |
| X-AnyLabel external path unchanged | 2 (explicit non-edit) |
| No soft-NMS / no train_detector auto-tile | honored (out of scope) |

Placeholder scan: cleared (Task 3 probe includes full temp-dataset implementation).

Type consistency: `get_vram_per_image`, `ensure_vram_estimates`, `iter_tile_windows`, `nms_xyxy`, `select_empty_tiles` names are stable across tasks.
