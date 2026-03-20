# YOLOv8 Toolkit

A clean, practical YOLOv8 training and detection toolkit that **automatically configures itself** based on your GPU, RAM, and dataset size — no manual tuning needed.

Built for real-world use: handles Windows paths, crash recovery, and scales from a laptop GPU to a workstation.

---

## Scripts

| Script | What it does |
|---|---|
| `train_detector.py` | Train a YOLOv8 detector on any labeled dataset |
| `detect_images.py` | Run a trained model on a folder of images, draw boxes + confidence |

---

## Requirements

```bash
pip install ultralytics opencv-python psutil
```

- Python 3.8+
- PyTorch with CUDA (for GPU training) — install from [pytorch.org](https://pytorch.org/get-started/locally/)
- Dataset in YOLOv8 format (e.g. exported from [Roboflow](https://roboflow.com))

---

## Training

### Basic usage
```bash
python train_detector.py --input /path/to/data.yaml --name my_detector_v1
```

### Windows paths work as-is
```bash
python train_detector.py --input c:\Users\Me\Dataset\data.yaml --name my_detector_v1
```

### Override any auto-calculated value
```bash
python train_detector.py --input c:\Users\Me\Dataset\data.yaml --name my_detector_v1 --model yolov8x.pt --batch 8
```

### Resume a crashed or interrupted run
```bash
# Resume specific run by name
python train_detector.py --resume --name my_detector_v1

# Resume most recent run automatically
python train_detector.py --resume
```

### All arguments
| Argument | Default | Description |
|---|---|---|
| `--input` | required | Path to `data.yaml` file |
| `--name` | `detector_v1` | Name for this training run |
| `--resume` | off | Resume from last checkpoint |
| `--model` | auto | Override model: `yolov8m/l/x.pt` |
| `--imgsz` | auto | Override image size in pixels |
| `--batch` | auto | Override batch size |
| `--workers` | auto | Override dataloader worker count |
| `--epochs` | `300` | Max training epochs |
| `--patience` | `50` | Early stopping patience |
| `--device` | `0` | `0` for GPU, `cpu` for CPU |

---

## Auto-Configuration

The script detects your hardware and dataset size, then selects the best model, image size, and batch size automatically.

### Model + image size selection

| Train images | VRAM | Model | imgsz | Reason |
|---|---|---|---|---|
| < 1,000 | ≥ 16GB | yolov8m | 1280 | Small dataset — m avoids overfit, 1280 fits |
| 1,000–5,000 | ≥ 16GB | yolov8l | 1024 | l+1280 forces batch 4 — too noisy |
| > 5,000 | ≥ 16GB | yolov8x | 1280 | Large dataset justifies x, 1280 fits at batch 8 |
| < 1,000 | ≥ 12GB | yolov8m | 1024 | Small dataset, safe imgsz |
| 1,000–5,000 | ≥ 12GB | yolov8l | 1024 | VRAM limit, keep imgsz safe |
| > 5,000 | ≥ 12GB | yolov8l | 1024 | l is safe ceiling at 12GB |
| < 1,000 | ≥ 8GB | yolov8m | 640 | Low VRAM, keep it safe |
| 1,000–5,000 | ≥ 8GB | yolov8m | 1024 | m is safe at 8GB + 1024 |
| > 5,000 | ≥ 8GB | yolov8l | 1024 | Dataset justifies l at safe imgsz |
| any | < 8GB | yolov8m | 640 | Low VRAM, minimal safe config |
| any | CPU | yolov8m | 640 | CPU fallback |

**Key constraint:** imgsz is only raised when batch size stays ≥ 8. Batch < 8 causes noisy gradients — `1024px + batch 8` beats `1280px + batch 4`.

### Batch size selection

Calculated from VRAM × 85% safety margin ÷ VRAM-per-image estimate, clamped to powers of 2 (4, 8, 16, 32, 64).

### Worker count selection

Calculated from CPU cores and RAM, capped at 8 for Windows stability.

You can always override any single value while letting the rest auto-calculate:
```bash
# Force x model, let batch and imgsz auto-calculate
python train_detector.py --input data.yaml --name run1 --model yolov8x.pt
```

---

## Output

Training results are saved to `runs/detect/<name>/`:

```
runs/detect/my_detector_v1/
├── weights/
│   ├── best.pt      <- best model weights (use this)
│   └── last.pt      <- last epoch checkpoint (used for resume)
├── results.png      <- loss and mAP curves
├── confusion_matrix.png
└── args.yaml        <- full training config
```

After training, the script automatically runs validation and reports:
```
[Metrics]
  mAP@0.50       : 0.8731  (target: > 0.80)
  mAP@0.50:0.95  : 0.6214

  PASS -- model meets target accuracy (mAP@0.50 >= 0.80)
```

---

## Detection

Run a trained model over a folder of images:

```bash
python detect_images.py --images /path/to/images --model runs/detect/my_detector_v1/weights/best.pt
```

### With custom confidence threshold
```bash
python detect_images.py --images /path/to/images --model best.pt --conf 0.5
```

### All arguments
| Argument | Default | Description |
|---|---|---|
| `--images` | required | Path to folder of input images |
| `--model` | required | Path to trained `.pt` model file |
| `--conf` | `0.25` | Minimum confidence threshold (0.0–1.0) |

Annotated images are saved to `<images_dir>/detections/` — originals are never modified.

Supported image formats: `.jpg` `.jpeg` `.png` `.bmp` `.tiff` `.tif` `.webp`

---

## Dataset Format

This toolkit expects datasets in **YOLOv8 format** with a `data.yaml` file:

```yaml
train: ../train/images
val:   ../valid/images
test:  ../test/images

nc: 1
names: ['my_object']
```

[Roboflow](https://roboflow.com) exports directly in this format — recommended for labeling and dataset management.

---

## Tips

- **mAP@0.50 < 0.80 after training?** Add more labeled images. Dataset size is the biggest factor in accuracy.
- **OOM crash?** Add `--batch 8` or `--batch 4` to override the auto-calculated batch size.
- **PC crashed mid-training?** Just run with `--resume` — picks up from the last saved epoch automatically.
- **Specific screw/fastener types not detected?** Collect photos from your actual target structure and fine-tune on those.

---

## License

MIT — use freely, no restrictions.
