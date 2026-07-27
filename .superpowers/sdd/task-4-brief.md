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

