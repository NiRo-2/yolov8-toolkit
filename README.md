# YOLOv8 Toolkit

A clean, practical YOLOv8 toolkit for building datasets, training detectors, and running inference — **automatically configures itself** based on your GPU, RAM, and dataset size — no manual tuning needed.

Built for real-world use: handles Windows paths, crash recovery, and scales from a laptop GPU to a workstation.

---

## Scripts

| Script | What it does |
|---|---|
| `vlm_yolo_prep.py` | Build a labelled YOLOv8 dataset from raw photos using a local VLM — no manual annotation needed |
| `voc_to_yolo.py` | Convert an existing Pascal VOC annotated dataset to YOLOv8 format |
| `train_detector.py` | Train a YOLOv8 detector on any labelled dataset |
| `detect_images.py` | Run a trained model on a folder of images, draw boxes + confidence |

---

## Full Pipeline

```
[Raw photos]                    [Existing VOC dataset]
     |                                   |
     v                                   v
vlm_yolo_prep.py            voc_to_yolo.py
(auto-label with VLM)       (convert annotations)
     |                                   |
     +-----------------------------------+
                     |
                     v
          Labelled YOLOv8 dataset
          (train / val + data.yaml)
                     |
                     v
          train_detector.py
                     |
                     v
          detect_images.py
```

---

## Requirements

```bash
pip install ultralytics opencv-python psutil requests pillow pyyaml
```

- Python 3.8+
- PyTorch with CUDA (for GPU training) — install from [pytorch.org](https://pytorch.org/get-started/locally/)
- [LM Studio](https://lmstudio.ai/) with a vision model loaded (for `vlm_yolo_prep.py` only)
- Dataset in YOLOv8 format (e.g. exported from [Roboflow](https://roboflow.com))

---

## Step 1a — Build a Dataset from Raw Photos (`vlm_yolo_prep.py`)

Sends each photo to a Vision-Language Model running locally in LM Studio, gets bounding box coordinates back, and writes a complete YOLOv8 dataset with no manual labelling.

### Basic usage
```bash
python vlm_yolo_prep.py \
    --input  C:/data/raw_photos \
    --output C:/data/dataset \
    --objects screw "hex bolt" "countersunk screw"
```

Classes are auto-assigned in the order you list them: `screw=0`, `hex bolt=1`, `countersunk screw=2`.

Annotated preview images are saved to `<output>/preview/` by default so you can visually verify detection quality before training.

### Windows batch file
```bat
python vlm_yolo_prep.py ^
    --input  "C:\data\raw_photos" ^
    --output "C:\data\dataset"    ^
    --objects Screw               ^
    --model   qwen2.5-vl-7b-instruct ^
    --confidence 0.9              ^
    --downsample 2
```

### LM Studio setup
1. Download and open [LM Studio](https://lmstudio.ai/)
2. Download one of these models:
   - **Best quality:** `Qwen2.5-VL-72B-Instruct-GGUF` Q4_K_M — needs 16 GB VRAM + 128 GB RAM, set context to **32k**
   - **Faster:** `Qwen2.5-VL-7B-Instruct-GGUF` Q8_0 — needs 8 GB VRAM, set context to **16k**
3. Load the model and start the local server (default port 1234)

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | required | Folder of raw images |
| `--output` | required | Output folder (created if absent, auto-versioned if not empty) |
| `--objects` | required | Object names to detect. Spaces allowed: `"hex bolt"` `"red car"` |
| `--classes` | auto | Override class IDs: `screw:0 bolt:0 "hex bolt":1`. Auto-assigned by default |
| `--model` | `qwen2.5-vl-72b-instruct` | Model name as shown in LM Studio |
| `--url` | `http://localhost:1234/...` | LM Studio API endpoint |
| `--timeout` | `180` | API timeout in seconds (use 60 for the 7B model) |
| `--retries` | `2` | Retry attempts per image on API failure |
| `--confidence` | `0.0` | Min detection confidence — `0.0` keeps all, `0.9` keeps only high-confidence |
| `--downsample` | `1.0` | Divide image dimensions by this factor before sending to VLM |
| `--train` | `0.70` | Train split ratio |
| `--val` | `0.20` | Val split ratio (gets remainder when `--enable-test` is off) |
| `--seed` | `42` | Random seed for reproducible splits |
| `--enable-test` | off | Create a test split in addition to train and val |
| `--no-preview` | off | Disable annotated preview image export |

### Output structure

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── preview/          ← annotated images for visual QA (one per input photo)
└── data.yaml
```

Generated `data.yaml`:
```yaml
train: ../train/images
val: ../val/images
nc: 1
names: ['screw']
```

> **Note:** VLM-generated labels are a strong starting point but not perfect. Review the `preview/` folder and correct any bad detections in [Roboflow](https://roboflow.com) or [LabelImg](https://github.com/HumanSignal/labelImg) before final training.

---

## Step 1b — Convert an Existing VOC Dataset (`voc_to_yolo.py`)

If you already have a dataset annotated in Pascal VOC format (XML files), this script converts it to YOLOv8 format in one command — no manual work needed.

### Basic usage
```bash
python voc_to_yolo.py --input C:/data/voc --output C:/data/dataset
```

Classes are auto-discovered from the XML files and sorted alphabetically. Use `--classes` to control the order (and therefore the class IDs) explicitly.

### Supported VOC layouts

The script handles all common VOC folder structures automatically:

```
# Flat — images and XMLs in the same folder
input/
  image1.jpg  image1.xml
  image2.jpg  image2.xml

# Standard VOC — separate subfolders
input/
  JPEGImages/   image1.jpg
  Annotations/  image1.xml

# Pre-split — already divided into train/val/test
input/
  train/  image1.jpg  image1.xml
  val/    image2.jpg  image2.xml
```

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | required | Folder containing VOC images and XML annotation files |
| `--output` | required | Output folder for the YOLOv8 dataset (created if absent, auto-versioned if not empty) |
| `--classes` | auto | Class names in order — controls ID assignment. Auto-discovered and sorted alphabetically if omitted |
| `--train` | `0.70` | Train split ratio |
| `--val` | `0.20` | Val split ratio (gets remainder when `--enable-test` is off) |
| `--seed` | `42` | Random seed for reproducible splits |
| `--enable-test` | off | Create a test split in addition to train and val |

### Examples

```bash
# Auto-discover classes, 70/30 split
python voc_to_yolo.py --input C:/data/voc --output C:/data/dataset

# Control class order (first = ID 0, second = ID 1, ...)
python voc_to_yolo.py --input C:/data/voc --output C:/data/dataset \
    --classes screw bolt "hex bolt"

# With test split
python voc_to_yolo.py --input C:/data/voc --output C:/data/dataset --enable-test
```

---

## Step 2 — Train (`train_detector.py`)

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

### Auto-configuration

The script detects your hardware and dataset size, then selects the best model, image size, and batch size automatically.

| Train images | VRAM | Model | imgsz | Reason |
|---|---|---|---|---|
| < 1,000 | ≥ 16GB | yolov8m | 1280 | Small dataset — m avoids overfit |
| 1,000–5,000 | ≥ 16GB | yolov8l | 1024 | l+1280 forces batch 4 — too noisy |
| > 5,000 | ≥ 16GB | yolov8x | 1280 | Large dataset justifies x |
| < 1,000 | ≥ 12GB | yolov8m | 1024 | Small dataset, safe imgsz |
| 1,000–5,000 | ≥ 12GB | yolov8l | 1024 | VRAM limit, keep imgsz safe |
| > 5,000 | ≥ 12GB | yolov8l | 1024 | l is safe ceiling at 12GB |
| < 1,000 | ≥ 8GB | yolov8m | 640 | Low VRAM, keep it safe |
| 1,000–5,000 | ≥ 8GB | yolov8m | 1024 | m is safe at 8GB + 1024 |
| > 5,000 | ≥ 8GB | yolov8l | 1024 | Dataset justifies l |
| any | < 8GB | yolov8m | 640 | Low VRAM, minimal safe config |
| any | CPU | yolov8m | 640 | CPU fallback |

**Key constraint:** imgsz is only raised when batch size stays ≥ 8. `1024px + batch 8` beats `1280px + batch 4`.

Batch size is calculated from VRAM × 85% safety margin ÷ VRAM-per-image estimate, clamped to powers of 2 (4, 8, 16, 32, 64). Worker count is calculated from CPU cores and RAM, capped at 8 for Windows stability.

You can always override any single value while letting the rest auto-calculate:
```bash
python train_detector.py --input data.yaml --name run1 --model yolov8x.pt
```

### Output

Training results are saved to `runs/detect/<name>/`:

```
runs/detect/my_detector_v1/
├── weights/
│   ├── best.pt      ← best model weights (use this)
│   └── last.pt      ← last epoch checkpoint (used for resume)
├── results.png
├── confusion_matrix.png
└── args.yaml
```

After training, the script automatically runs validation and reports:
```
[Metrics]
  mAP@0.50       : 0.8731  (target: > 0.80)
  mAP@0.50:0.95  : 0.6214

  PASS -- model meets target accuracy (mAP@0.50 >= 0.80)
```

---

## Step 3 — Detect (`detect_images.py`)

Run a trained model over a folder of images:

```bash
python detect_images.py \
    --images /path/to/images \
    --model  runs/detect/my_detector_v1/weights/best.pt
```

### With custom confidence threshold
```bash
python detect_images.py --images /path/to/images --model best.pt --conf 0.5
```

### Export detection data as JSON
```bash
python detect_images.py --images /path/to/images --model best.pt --export-json
```

Saves `<images_dir>/detections/<name>.json` per image with detections, plus `labels.txt` mapping class IDs to names.

JSON format per detection:
```json
{
  "class_id": 0,
  "class_name": "screw",
  "confidence": 0.95,
  "pixel": {"x1": 120, "y1": 80, "x2": 340, "y2": 520},
  "yolo": {"cx": 0.273438, "cy": 0.317188, "bw": 0.21875, "bh": 0.4375}
}
```

- `pixel` — absolute (x1, y1, x2, y2) in image coordinates
- `yolo` — normalized (cx, cy, bw, bh) / (W, H) for YOLO training/export

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--images` | required | Path to folder of input images |
| `--model` | required | Path to trained `.pt` model file |
| `--conf` | `0.25` | Minimum confidence threshold (0.0–1.0) |
| `--export-json` | `True` | Export detection JSON + labels.txt in detections/ |

Annotated images are saved to `<images_dir>/detections/` — originals are never modified.
Output images preserve all original EXIF metadata (GPS, camera model, sensor data, etc.).

Supported formats: `.jpg` `.jpeg` `.png` `.bmp` `.tiff` `.tif` `.webp`

---

## Dataset Format

All scripts produce and consume datasets in **YOLOv8 format** with a `data.yaml` file:

```yaml
train: ../train/images
val: ../val/images

nc: 1
names: ['my_object']
```

Both `vlm_yolo_prep.py` and `voc_to_yolo.py` generate this file automatically. [Roboflow](https://roboflow.com) also exports directly in this format and is recommended for manual labelling and dataset management.

---

## Tips

- **Poor detections from the VLM?** Lower `--confidence` to `0.5` and review the `preview/` folder. The VLM may need a more descriptive object name.
- **VOC class names don't match what you want?** Use `--classes` to rename or reorder them when converting.
- **mAP@0.50 < 0.80 after training?** Add more labelled images — dataset size is the biggest factor in accuracy.
- **OOM crash during training?** Add `--batch 8` or `--batch 4` to override the auto-calculated batch size.
- **PC crashed mid-training?** Run with `--resume` — picks up from the last saved epoch automatically.
- **Specific variants not detected?** (e.g. rusted bolts, painted-over screws) Collect photos of those specific types and add them to the dataset.

---

## License

MIT — use freely, no restrictions.
