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

