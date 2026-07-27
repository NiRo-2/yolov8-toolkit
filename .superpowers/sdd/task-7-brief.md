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
