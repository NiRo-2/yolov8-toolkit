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

