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

